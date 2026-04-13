FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY go_renderer/ go_renderer/
RUN go build -o /renderer go_renderer/cmd/renderer/main.go

FROM alpine:3.19
RUN adduser -D -s /bin/false appuser
WORKDIR /app
COPY --from=builder /renderer /app/renderer
COPY go_renderer/web/ /app/go_renderer/web/
RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
CMD ["/app/renderer", "-port=8080", "-data=/app/data"]
