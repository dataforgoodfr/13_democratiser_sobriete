# Chat Sufficiency Frontend

SvelteKit frontend for Chat Sufficiency. It proxies chat/feedback calls to the FastAPI backend and uses Auth.js with Keycloak for SSO.

## Prerequisites

- Node.js 20+
- A Keycloak OpenID Connect client (confidential)

## Environment variables

Copy `.env.example` to `.env` and set values:

- `CHAT_SUFFICIENCY_API_URL`: backend base URL (e.g. `http://localhost:8000`)
- `PUBLIC_SHOW_FULL_NAVBAR`: `true` or `false`
- `AUTH_SECRET`: random secret (at least 32 chars)
- `AUTH_TRUST_HOST`: `true` on deployments behind a proxy
- `AUTH_KEYCLOAK_ID`: Keycloak client ID
- `AUTH_KEYCLOAK_SECRET`: Keycloak client secret
- `AUTH_KEYCLOAK_ISSUER`: realm issuer URL (must include realm)

Example issuer:

`https://keycloak.example.org/realms/your-realm`

## Keycloak client settings

Use a **confidential** OIDC client with Standard Flow enabled.

- Valid redirect URIs:
	- `http://localhost:5173/auth/callback/keycloak`
	- `https://chat.thesufficiencylab.org/auth/callback/keycloak`
- Valid post logout redirect URIs:
	- `http://localhost:5173`
	- `https://chat.thesufficiencylab.org`
- Web origins:
	- `http://localhost:5173`
	- `https://chat.thesufficiencylab.org`

If you want end users to self-register ("sign-up"), enable user registration in the Keycloak realm.

## Run locally

```sh
npm install
npm run dev
```

## Build

```sh
npm run build
npm run start
```
