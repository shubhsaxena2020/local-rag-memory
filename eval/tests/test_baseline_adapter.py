import importlib
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import baseline_adapter as ba  # noqa: E402

URL = "http://127.0.0.1:8097"
KEY = "SECRET-KEY-123"
KW = dict(base_url=URL, tenant_id="bench_t1", key=KEY)


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake(payload):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return mock.patch.object(ba._OPENER, "open", return_value=FakeResp(raw))


class UrlTests(unittest.TestCase):
    def test_accepts_loopback(self):
        self.assertEqual(ba.validate_base_url("http://127.0.0.1:8097/"), URL)
        self.assertEqual(ba.validate_base_url("http://[::1]:9000"), "http://[::1]:9000")

    def test_rejects_bad(self):
        for u in ["https://127.0.0.1:8097", "http://localhost:8097", "http://example.com:8097",
                  "http://10.0.0.1:8097", "http://127.0.0.1", "http://127.0.0.1:80",
                  "http://127.0.0.1:99999", "http://u:p@127.0.0.1:8097", "http://127.0.0.1:8097/x",
                  "http://127.0.0.1:8097/?a=1", "http://127.0.0.1.evil.com:8097", "", None, 5]:
            with self.subTest(u=u), self.assertRaises(ValueError):
                ba.validate_base_url(u)


class RecallTests(unittest.TestCase):
    def test_success_and_request_shape(self):
        with fake({"results": [{"id": 1}, {"id": 2}, {"id": 3}]}) as m:
            self.assertEqual(ba.memory_recall("q", 2, **KW), [{"id": 1}, {"id": 2}])
        req = m.call_args.args[0]
        self.assertEqual(req.full_url, f"{URL}/api/v1/bench_t1/query")
        self.assertEqual(req.get_header("Authorization"), f"Bearer {KEY}")
        self.assertEqual(json.loads(req.data), {"question": "q", "top_k": 2})
        self.assertEqual(m.call_args.kwargs["timeout"], 5.0)

    def test_input_validation_makes_no_call(self):
        bad = [dict(question=" "), dict(question="x" * 4001), dict(top_k=0), dict(top_k=True),
               dict(top_k=101), dict(timeout=0), dict(timeout=31), dict(timeout=True),
               dict(tenant_id="a/b"), dict(tenant_id=""), dict(tenant_id="../x"),
               dict(key=""), dict(key="a b"), dict(key="k\r\nX: y"),
               dict(base_url="http://8.8.8.8:8097")]
        for over in bad:
            args = {"question": "q", "top_k": 5, **KW, **over}
            with self.subTest(over=over), mock.patch.object(ba._OPENER, "open") as m:
                with self.assertRaises(ValueError):
                    ba.memory_recall(args.pop("question"), args.pop("top_k"), **args)
                m.assert_not_called()

    def test_errors_are_safe(self):
        errs = [urllib.error.URLError(f"boom {KEY} {URL}"), TimeoutError(KEY),
                ConnectionRefusedError(KEY),
                urllib.error.HTTPError(f"{URL}/{KEY}", 401, KEY, {}, io.BytesIO(b""))]
        for e in errs:
            with self.subTest(e=type(e).__name__), \
                    mock.patch.object(ba._OPENER, "open", side_effect=e):
                with self.assertRaises(ba.BaselineError) as cm:
                    ba.memory_recall("q", **KW)
                self.assertNotIn(KEY, str(cm.exception))
                self.assertNotIn("bench_t1", str(cm.exception))
                self.assertIsNone(cm.exception.__cause__)
                self.assertTrue(cm.exception.__suppress_context__)

    def test_response_validation(self):
        for payload in [b"not json", [], {"results": "x"}, {}, {"results": [1]},
                        {"results": None}, b"x" * (ba.MAX_RESPONSE_BYTES + 1)]:
            with self.subTest(p=str(payload)[:20]), fake(payload):
                with self.assertRaises(ba.BaselineError):
                    ba.memory_recall("q", **KW)

    def test_no_redirect_no_proxy(self):
        # An empty ProxyHandler registers no handlers, and it suppresses the
        # default env-driven one, so no proxy handler may be active.
        self.assertFalse(any(isinstance(x, ba.urllib.request.ProxyHandler)
                             for x in ba._OPENER.handlers))
        self.assertIn(ba._NoRedirect, {type(x) for x in ba._OPENER.handlers})
        self.assertIsNone(
            ba._NoRedirect().redirect_request(None, None, 302, "", {}, "http://evil/"))


class ImportTests(unittest.TestCase):
    def test_import_has_no_credentials_or_network(self):
        with mock.patch("socket.socket", side_effect=AssertionError("network at import")), \
                mock.patch.dict("os.environ", {}, clear=True):
            importlib.reload(ba)
        self.assertFalse(any(n in vars(ba) for n in ("TENANT_KEY", "TENANT_ID", "BASE_URL")))


if __name__ == "__main__":
    unittest.main()
