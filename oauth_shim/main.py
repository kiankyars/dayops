from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="dayops-oauth-shim", version="0.1.0")

OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar",
]


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _client_config() -> dict[str, Any]:
    return {
        "web": {
            "client_id": _env("GOOGLE_OAUTH_CLIENT_ID"),
            "client_secret": _env("GOOGLE_OAUTH_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


app.add_middleware(SessionMiddleware, secret_key=_env("SHIM_SESSION_SECRET"))


def _fetch_google_userinfo(access_token: str) -> dict[str, Any]:
    req = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"google_userinfo_failed: {exc.code}") from exc


def _post_bootstrap(userinfo: dict[str, Any], token_json: str) -> str:
    body = json.dumps({"userinfo": userinfo, "token_json": token_json}).encode("utf-8")
    req = urllib.request.Request(
        _env("DAYOPS_BOOTSTRAP_URL"),
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-bootstrap-secret": _env("DAYOPS_OAUTH_BOOTSTRAP_SECRET"),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
            auth_token = str(payload.get("auth_token", "")).strip()
            if not auth_token:
                raise HTTPException(status_code=502, detail="dayops_bootstrap_missing_auth_token")
            return auth_token
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"dayops_bootstrap_failed: {exc.code}") from exc


@app.get("/auth/google/start")
def auth_google_start(request: Request) -> RedirectResponse:
    flow = Flow.from_client_config(_client_config(), scopes=OAUTH_SCOPES, autogenerate_code_verifier=True)
    flow.redirect_uri = _env("GOOGLE_OAUTH_REDIRECT_URI")
    auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
    request.session["oauth_state"] = state
    request.session["oauth_code_verifier"] = flow.code_verifier
    request.session["redirect_after_auth"] = request.query_params.get("next", "").strip() or _env("DAYOPS_APP_URL")
    return RedirectResponse(auth_url, status_code=302)


@app.get("/auth/google/callback")
def auth_google_callback(request: Request) -> RedirectResponse:
    state = str(request.session.get("oauth_state", "")).strip()
    code_verifier = str(request.session.get("oauth_code_verifier", "")).strip()
    if not state:
        raise HTTPException(status_code=400, detail="oauth_state_missing")
    if not code_verifier:
        raise HTTPException(status_code=400, detail="oauth_code_verifier_missing")

    flow = Flow.from_client_config(
        _client_config(),
        scopes=OAUTH_SCOPES,
        state=state,
    )
    redirect_uri = _env("GOOGLE_OAUTH_REDIRECT_URI")
    flow.redirect_uri = redirect_uri
    flow.code_verifier = code_verifier
    # Behind Cloud Run, request.url may be observed as http internally.
    authorization_response = f"{redirect_uri}?{request.url.query}"
    flow.fetch_token(authorization_response=authorization_response)

    userinfo = _fetch_google_userinfo(flow.credentials.token)
    auth_token = _post_bootstrap(userinfo, flow.credentials.to_json())
    request.session.pop("oauth_state", None)
    request.session.pop("oauth_code_verifier", None)
    request.session.pop("redirect_after_auth", None)
    login_url = _env("DAYOPS_AUTH_COMPLETE_URL") + f"?token={urllib.parse.quote(auth_token, safe='')}"
    return RedirectResponse(login_url, status_code=302)
