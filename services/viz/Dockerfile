FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_WS_URL=ws://localhost:8765
ARG VITE_DEFAULT_SOURCE=live
ENV VITE_WS_URL=$VITE_WS_URL
ENV VITE_DEFAULT_SOURCE=$VITE_DEFAULT_SOURCE

RUN npm run build

FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
