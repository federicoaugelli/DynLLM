#!/usr/bin/env bash
# DynLLM systemd service installer
# Run as root: sudo bash systemd/install.sh

set -euo pipefail

INSTALL_DIR="/opt/dynllm"
SERVICE_FILE="$(dirname "$0")/dynllm.service"
SERVICE_DEST="/etc/systemd/system/dynllm.service"
DYNLLM_USER="dynllm"
DYNLLM_GROUP="dynllm"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || error "This script must be run as root (sudo)."
}

check_dependency() {
    command -v "$1" &>/dev/null || error "'$1' is required but not found on PATH."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

require_root
check_dependency uv
check_dependency systemctl

info "Creating system user/group '$DYNLLM_USER' (if not exists)..."
id -u "$DYNLLM_USER" &>/dev/null || \
    useradd --system --no-create-home --shell /sbin/nologin \
            --comment "DynLLM service account" \
            "$DYNLLM_USER"

info "Setting up installation directory at $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"

# Copy project files if running from project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    info "Copying project files from $SCRIPT_DIR to $INSTALL_DIR ..."
    cp -r "$SCRIPT_DIR"/. "$INSTALL_DIR/"
else
    warn "Could not find pyproject.toml. Make sure to manually place your " \
         "project files in $INSTALL_DIR and create a config.yaml there."
fi

info "Installing Python dependencies with uv ..."
(cd "$INSTALL_DIR" && uv sync --frozen 2>&1 | tail -5)

info "Creating a default config.yaml (if not present)..."
if [[ ! -f "$INSTALL_DIR/config.yaml" ]]; then
    cp "$INSTALL_DIR/config.example.yaml" "$INSTALL_DIR/config.yaml" 2>/dev/null || \
    cat > "$INSTALL_DIR/config.yaml" <<'EOF'
# DynLLM configuration – edit before starting the service.
# See config.example.yaml for full documentation.

server:
  host: "0.0.0.0"
  port: 8000

total_vram_mb: 8192
idle_timeout_seconds: 300

enabled_backends:
  - llamacpp
  - openvino

models: []
EOF
    info "Created default $INSTALL_DIR/config.yaml – please edit it."
fi

info "Setting ownership of $INSTALL_DIR ..."
chown -R "$DYNLLM_USER:$DYNLLM_GROUP" "$INSTALL_DIR"

info "Installing systemd service file to $SERVICE_DEST ..."
cp "$SERVICE_FILE" "$SERVICE_DEST"

# Patch WorkingDirectory and ExecStart to actual install dir
sed -i "s|/opt/dynllm|$INSTALL_DIR|g" "$SERVICE_DEST"

info "Reloading systemd daemon ..."
systemctl daemon-reload

info "Enabling dynllm.service ..."
systemctl enable dynllm.service

cat <<'INSTRUCTIONS'

Installation complete!

Next steps:
  1. Edit /opt/dynllm/config.yaml to configure your models and VRAM budget.
  2. Start the service:
       sudo systemctl start dynllm
  3. Check logs:
       sudo journalctl -u dynllm -f
  4. To stop:
       sudo systemctl stop dynllm

INSTRUCTIONS
