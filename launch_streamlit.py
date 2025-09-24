import sys, os
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # ensure it doesn't complain about dev mode + port combo
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    sys.argv = [
        "streamlit", "run", "streamlit_app.py",
        "--browser.gatherUsageStats=false",
        "--server.headless=true",
        # no --server.port here
    ]
    raise SystemExit(stcli.main())