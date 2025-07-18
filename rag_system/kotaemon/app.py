import os
import logging
import sys

from theflow.settings import settings as flowsettings

KH_APP_DATA_DIR = getattr(flowsettings, "KH_APP_DATA_DIR", ".")
KH_GRADIO_SHARE = getattr(flowsettings, "KH_GRADIO_SHARE", False)
GRADIO_TEMP_DIR = os.getenv("GRADIO_TEMP_DIR", None)
GRADIO_SERVER_PORT = os.getenv("GRADIO_SERVER_PORT", "7860")
# override GRADIO_TEMP_DIR if it's not set
if GRADIO_TEMP_DIR is None:
    GRADIO_TEMP_DIR = os.path.join(KH_APP_DATA_DIR, "gradio_tmp")
    os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

# Configure logging for Docker Swarm visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting Kotaemon application...")

from ktem.main import App  # noqa

app = App()
demo = app.make()

logger.info(f"Launching Gradio app on port {GRADIO_SERVER_PORT}")

demo.queue().launch(
    favicon_path=app._favicon,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=int(GRADIO_SERVER_PORT),
    allowed_paths=[
        "libs/ktem/ktem/assets",
        GRADIO_TEMP_DIR,
    ],
    share=KH_GRADIO_SHARE,
    show_error=True,  # Show errors in logs
    quiet=False,  # Don't suppress logs
)
