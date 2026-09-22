Alpha2048 - Apple Silicon macOS CPU test build

Applications
------------
Alpha2048.app
  The 2048 game and in-game AI assistant.

Alpha2048ScreenAssistant.app
  Read-only recommendations for a visible external 2048 board.

Opening the apps
----------------
This test build is not notarized. If macOS blocks the first launch, right-click
the app in Finder, choose Open, and confirm Open. Do not remove files from an
app bundle.

The screen assistant needs Screen Recording permission. macOS should prompt on
first use. You can also enable it in System Settings > Privacy & Security >
Screen Recording, then quit and reopen the assistant.

This package uses CPU inference. CUDA and Apple MPS are not included.

Logs are written under ~/Library/Logs/Alpha2048 when possible.
