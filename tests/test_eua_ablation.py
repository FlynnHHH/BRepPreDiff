import torch
import pytest
from brepprediff.models.encoder import EdgeAttentionLayer


def layer(edge='full', attention='softmax'):
    torch.manual_seed(42)
    return EdgeAttentionLayer(16, 0.0, num_heads=4, update_edges=True,
                              edge_update_mode=edge, attention_mode=attention)


def inputs():
    torch.manual_seed(7)
    return torch.randn(4,16), torch.randn(5,16), torch.tensor([[0,1,2,0,3],[1,2,1,2,0]])


def test_fixed_edges_keep_values_and_embedding_gradients():
    m=layer('fixed')
    x,e,idx=inputs(); e.requires_grad_()
    for _ in range(3):
        x,out=m(x,e,idx)
        assert out is e
    x.square().sum().backward()
    assert e.grad is not None and e.grad.abs().sum()>0
    assert m.edge_update.net[0].weight.grad is None


def test_edge_only_independent_of_faces_and_endpoint_weights():
    m=layer('edge_only'); x,e,idx=inputs()
    _,a=m(x,e,idx); _,b=m(x*7+3,e,idx)
    torch.testing.assert_close(a,b)
    a[:,0].sum().backward()
    grad=m.edge_update.net[0].weight.grad
    assert torch.count_nonzero(grad[:,:32])==0
    assert grad[:,32:].abs().sum()>0
    assert not torch.allclose(a,e)


def test_uniform_matches_manual_mean_and_ignores_query_key_bias():
    m=layer('fixed','uniform'); x,e,idx=inputs(); src,dst=idx
    norm=m.norm_attention(x)
    values=m.value(norm)[src]+m.edge_value(e)
    mean=torch.zeros_like(x)
    for n in range(len(x)):
        if (dst==n).any(): mean[n]=values[dst==n].mean(0)
    expected=x+m.output_projection(mean)
    expected=expected+m.ffn(m.norm_ffn(expected))
    actual,_=m(x,e,idx)
    torch.testing.assert_close(actual,expected)
    with torch.no_grad():
        for module in [m.query,m.key,m.edge_key,m.edge_bias]:
            for p in module.parameters(): p.fill_(100)
    torch.testing.assert_close(m(x,e,idx)[0],actual)
    actual.sum().backward()
    assert m.query.weight.grad is None
    assert m.edge_value.weight.grad is not None


@pytest.mark.parametrize('edge,attention',[('full','softmax'),('fixed','softmax'),('edge_only','softmax'),('full','uniform'),('fixed','uniform')])
def test_modes_preserve_initialization_and_handle_empty_graph(edge,attention):
    base=layer(); m=layer(edge,attention)
    for k,v in base.state_dict().items(): torch.testing.assert_close(m.state_dict()[k],v)
    x,e,idx=inputs()
    out,edges=m(x,e[:0],idx[:,:0])
    assert torch.isfinite(out).all() and edges.shape==(0,16)
    out.sum().backward()


def test_full_mode_matches_legacy_forward():
    m=layer(); x,e,idx=inputs(); src,dst=idx
    from brepprediff.models.encoder import segmented_softmax
    n=m.norm_attention(x)
    q=m.query(n).view(-1,4,4)[dst]
    k=(m.key(n)[src]+m.edge_key(e)).view(-1,4,4)
    v=(m.value(n)[src]+m.edge_value(e)).view(-1,4,4)
    w=segmented_softmax((q*k).sum(-1)/2+m.edge_bias(e),dst,len(x))
    agg=x.new_zeros(len(x),4,4).index_add_(0,dst,v*w.unsqueeze(-1))
    expected=x+m.output_projection(agg.flatten(1))
    expected=expected+m.ffn(m.norm_ffn(expected))
    edge_expected=m.edge_norm(e+m.edge_update(torch.cat([expected[src],expected[dst],e],-1)))
    actual,edge_actual=m(x,e,idx)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(edge_actual,edge_expected,rtol=0,atol=0)
