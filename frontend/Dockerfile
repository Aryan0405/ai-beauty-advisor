# Build context for this Dockerfile is the frontend/ directory itself (see
# docker-compose.yml).

# ---- Build stage: compile the Vite production bundle ----
FROM node:22-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Vite inlines VITE_* variables into the static bundle at build time, so it
# must be supplied as a build arg, not just a runtime container env var.
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# ---- Runtime stage: serve the static bundle with nginx, non-root ----
FROM nginx:alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf

# The stock image starts nginx's master process as root so it can bind to
# port 80 and manage worker processes. Reconfigure it to listen on an
# unprivileged port and drop root entirely (spec section 17).
RUN sed -i '/^user /d' /etc/nginx/nginx.conf \
    && mkdir -p /var/cache/nginx/client_temp \
       /var/cache/nginx/proxy_temp \
       /var/cache/nginx/fastcgi_temp \
       /var/cache/nginx/uwsgi_temp \
       /var/cache/nginx/scgi_temp \
    && touch /var/run/nginx.pid \
    && chown -R nginx:nginx /var/cache/nginx /var/run/nginx.pid /usr/share/nginx/html /etc/nginx/conf.d

COPY --from=builder /app/dist /usr/share/nginx/html

USER nginx
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
