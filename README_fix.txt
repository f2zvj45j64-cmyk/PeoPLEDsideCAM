Web trigger return fix:
- FastAPI endpoint now waits with asyncio.sleep (no invalid await).
- Camera open() returns correctly.
- GPIO start logic fixed.
- Manual capture overlay save fixed.

Run:
  python3 JetsonDualCamPy.py

Web:
  POST /api/eval
