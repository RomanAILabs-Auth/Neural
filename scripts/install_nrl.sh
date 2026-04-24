#!/usr/bin/env bash
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
set -euo pipefail

OPT_IN="${NRL_INSTALL_OPT_IN_LM_AI:-ask}"
SKIP_BUILD="${NRL_INSTALL_SKIP_BUILD:-0}"
NO_PATH_UPDATE="${NRL_INSTALL_NO_PATH_UPDATE:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_BIN="${HOME}/.local/bin"
CONSENT_DIR="${HOME}/.nrl"
CONSENT_PATH="${CONSENT_DIR}/consent.json"
NRL_BIN_SRC="${ROOT}/build/bin/nrl"

if [ "$SKIP_BUILD" != "1" ]; then
  echo "[install] building NRL release artifacts ..."
  "$ROOT/build.sh"
fi

if [ ! -f "$NRL_BIN_SRC" ]; then
  echo "missing nrl binary at $NRL_BIN_SRC" >&2
  exit 1
fi

mkdir -p "$INSTALL_BIN"
cp "$NRL_BIN_SRC" "${INSTALL_BIN}/nrl"
chmod +x "${INSTALL_BIN}/nrl"

# Demo + nrlpy runtime for ``nrl demo`` (PYTHONPATH=$NRL_ROOT/py)
SHARE="${HOME}/.local/share/nrl"
mkdir -p "${SHARE}/build/bin"
cp -f "${INSTALL_BIN}/nrl" "${SHARE}/build/bin/nrl"
chmod +x "${SHARE}/build/bin/nrl"

# ``nrlpy`` on PATH (TriPy-style): thin wrapper around ``python3 -m nrlpy.cli``
cat >"${INSTALL_BIN}/nrlpy" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export NRL_ROOT="${NRL_ROOT:-${HOME}/.local/share/nrl}"
if command -v nrl >/dev/null 2>&1; then
  export NRL_BIN="$(command -v nrl)"
fi
export PYTHONPATH="${NRL_ROOT}/py${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m nrlpy.cli "$@"
EOF
chmod +x "${INSTALL_BIN}/nrlpy"

mkdir -p "${SHARE}/examples" "${SHARE}/py/nrlpy"
cp -f "${ROOT}/examples/"*.py "${SHARE}/examples/" 2>/dev/null || true
cp -Rf "${ROOT}/nrlpy/src/nrlpy/"* "${SHARE}/py/nrlpy/"
export NRL_ROOT="${SHARE}"

if [ "$NO_PATH_UPDATE" != "1" ]; then
  case ":${PATH}:" in
    *":${INSTALL_BIN}:"*) ;;
    *)
      SHELL_RC="${HOME}/.bashrc"
      if [ -n "${ZSH_VERSION:-}" ]; then
        SHELL_RC="${HOME}/.zshrc"
      fi
      if [ -f "$SHELL_RC" ] && ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$SHELL_RC"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
      fi
      export PATH="${INSTALL_BIN}:${PATH}"
      ;;
  esac
fi

mkdir -p "$CONSENT_DIR"
ENABLE="false"
if [ "$OPT_IN" = "1" ] || [ "$OPT_IN" = "true" ]; then
  ENABLE="true"
elif [ "$OPT_IN" = "0" ] || [ "$OPT_IN" = "false" ]; then
  ENABLE="false"
else
  read -r -p "Opt in to LM/AI features? (y/N) " REPLY
  case "$REPLY" in
    y|Y|yes|YES) ENABLE="true" ;;
    *) ENABLE="false" ;;
  esac
fi

if [ "$ENABLE" = "true" ]; then
  export NRL_LM_AI_OPT_IN=1
else
  export NRL_LM_AI_OPT_IN=0
fi

cat > "$CONSENT_PATH" <<EOF
{
  "lm_ai_opt_in": $ENABLE,
  "source": "install_nrl.sh",
  "updated_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

SHELL_RC="${HOME}/.bashrc"
if [ -n "${ZSH_VERSION:-}" ]; then
  SHELL_RC="${HOME}/.zshrc"
fi
if [ -f "$SHELL_RC" ] && ! grep -q 'export NRL_ROOT=' "$SHELL_RC" 2>/dev/null; then
  printf '\n# NRL demo / runtime root\nexport NRL_ROOT="%s"\n' "${SHARE}" >>"$SHELL_RC"
fi

echo "[install] complete"
echo "  binary:  ${INSTALL_BIN}/nrl"
echo "  nrlpy:   ${INSTALL_BIN}/nrlpy  (reload shell, then: nrlpy script.py)"
echo "  consent: ${CONSENT_PATH}"
echo "  lm_ai_opt_in: ${ENABLE}"
echo "  NRL_ROOT: ${SHARE} (examples + py/nrlpy for nrl demo)"
echo "  health:  nrlpy doctor"
echo "  quick:   nrlpy absorb <model.gguf>; nrlpy chat <model.gguf> --rewired"
