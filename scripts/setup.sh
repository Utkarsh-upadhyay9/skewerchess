#!/usr/bin/env bash
# skewerchess one-shot setup script.
# Installs everything you need on a fresh macOS (Apple Silicon).
# You will be prompted for your Mac password ONCE (when Homebrew installs).

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

step() { echo -e "\n${BOLD}${GREEN}==>${NC} ${BOLD}$1${NC}"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
ok()   { echo -e "${GREEN}✓${NC} $1"; }

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
  fail "This script is for Apple Silicon Macs only. Detected: $ARCH"
fi

step "1/8 Xcode Command Line Tools"
if xcode-select -p >/dev/null 2>&1; then
  ok "already installed: $(xcode-select -p)"
else
  warn "installing — a GUI popup will appear, click Install and wait"
  xcode-select --install || true
  echo "Press Enter once the GUI installer has finished..."
  read -r
fi

step "2/8 Homebrew"
if command -v brew >/dev/null 2>&1; then
  ok "already installed: $(brew --version | head -1)"
else
  warn "installing Homebrew — you'll be prompted for your Mac password"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    {
      echo ''
      echo '# Homebrew'
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"'
    } >> "$HOME/.zprofile"
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
  ok "Homebrew installed"
fi

step "3/8 Homebrew packages"
PKGS=(git gh stockfish uv fnm pnpm jq wget htop)
for pkg in "${PKGS[@]}"; do
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    ok "$pkg already installed"
  else
    echo "installing $pkg..."
    brew install "$pkg"
  fi
done

step "4/8 Configure zsh"
ZSHRC="$HOME/.zshrc"
touch "$ZSHRC"

if ! grep -q 'setopt interactive_comments' "$ZSHRC"; then
  echo '' >> "$ZSHRC"
  echo '# Allow `#` comments when pasting commands' >> "$ZSHRC"
  echo 'setopt interactive_comments' >> "$ZSHRC"
  ok "added interactive_comments to .zshrc"
else
  ok "interactive_comments already set"
fi

if ! grep -q 'fnm env' "$ZSHRC"; then
  echo '' >> "$ZSHRC"
  echo '# fnm (Node version manager)' >> "$ZSHRC"
  echo 'eval "$(fnm env --use-on-cd --shell zsh)"' >> "$ZSHRC"
  ok "added fnm to .zshrc"
else
  ok "fnm already in .zshrc"
fi

eval "$(fnm env --shell bash)" 2>/dev/null || true

step "5/8 Python 3.12 (via uv)"
uv python install 3.12
ok "Python 3.12 installed via uv"

step "6/8 Node LTS (via fnm)"
fnm install --lts
fnm default lts-latest
ok "Node $(fnm exec --using=lts-latest -- node --version) installed"

step "7/8 Project virtualenv + Python deps"
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then
  uv venv
  ok "created .venv"
fi
uv sync --extra dev
ok "Python deps installed"

step "8/8 Verifying everything"
echo ""
echo "Tool versions:"
printf "  brew      %s\n" "$(brew --version | head -1)"
printf "  git       %s\n" "$(git --version)"
printf "  gh        %s\n" "$(gh --version | head -1)"
printf "  stockfish %s\n" "$(echo "uci" | stockfish 2>/dev/null | head -1)"
printf "  uv        %s\n" "$(uv --version)"
printf "  fnm       %s\n" "$(fnm --version)"
printf "  pnpm      %s\n" "$(pnpm --version)"
printf "  python    %s\n" "$(uv run python --version)"

echo ""
ok "Setup complete!"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  1. Restart your terminal (or run: ${YELLOW}source ~/.zshrc${NC})"
echo "  2. Set your git identity:"
echo "       ${YELLOW}git config --global user.name \"Your Name\"${NC}"
echo "       ${YELLOW}git config --global user.email \"you@example.com\"${NC}"
echo "  3. Authenticate GitHub: ${YELLOW}gh auth login${NC}"
echo "  4. Copy env template:   ${YELLOW}cp .env.example .env${NC}  and fill in your keys"
echo "  5. Run smoke test:      ${YELLOW}uv run python scripts/smoke_test.py${NC}"
