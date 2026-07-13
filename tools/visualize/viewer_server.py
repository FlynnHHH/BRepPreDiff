from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import json


ROOT = Path(__file__).resolve().parent
FAVORITES_FILE = ROOT / "favorites.json"
RESULTS_DIR = ROOT / "results"
INSTANCE_SUFFIX = "_instance_pred_rgb.ply"
SEMANTIC_SUFFIX = "_semantic_pred.ply"


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        request_path = urlparse(self.path).path.rstrip("/")
        if request_path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if request_path == "/api/favorites":
            self.send_json(load_favorites())
            return
        if request_path == "/api/models":
            self.send_json({"files": get_model_files()})
            return
        super().do_GET()

    def do_POST(self):
        request_path = urlparse(self.path).path.rstrip("/")
        if request_path != "/api/favorites":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body or "{}")
            favorites = data.get("favorites", [])
            if not isinstance(favorites, list):
                raise ValueError("favorites must be a list")
            save_favorites(favorites)
            self.send_json({"ok": True, "favorites": sanitize_favorites(favorites)})
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)

    def send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def load_favorites():
    if not FAVORITES_FILE.exists():
        return {"favorites": []}
    try:
        data = json.loads(FAVORITES_FILE.read_text(encoding="utf-8"))
        favorites = data.get("favorites", [])
        if not isinstance(favorites, list):
            favorites = []
        return {"favorites": sanitize_favorites(favorites)}
    except Exception:
        return {"favorites": []}


def save_favorites(favorites):
    data = {"favorites": sanitize_favorites(favorites)}
    FAVORITES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_favorite_name(name):
    if not isinstance(name, str):
        return None

    if name.endswith("_instance_colors_pred.ply"):
        return name[: -len("_instance_colors_pred.ply")]

    if name.endswith("_instance_pred_rgb.ply"):
        return name[: -len("_instance_pred_rgb.ply")]

    if name.endswith("_semantic_pred.ply"):
        return name[: -len("_semantic_pred.ply")]

    if name.endswith("_pred.ply"):
        name = name[:-9]
    return name


def sanitize_favorites(raw_favorites):
    valid = set(get_sample_names())
    normalized = [
        normalize_favorite_name(item)
        for item in raw_favorites
        if isinstance(item, str)
    ]
    return sorted(valid.intersection(set(filter(None, normalized))))


def get_sample_names():
    names = []
    for name in get_model_files():
        normalized = normalize_favorite_name(name)
        if normalized:
            names.append(normalized)
    return sorted(set(names))


def get_model_files():
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [
            item.name
            for item in RESULTS_DIR.iterdir()
            if item.is_file() and item.name.endswith(".ply")
        ]
    )


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8061), ViewerHandler)
    print("PLY viewer: http://localhost:8061/")
    print(f"Root: {ROOT}")
    server.serve_forever()
