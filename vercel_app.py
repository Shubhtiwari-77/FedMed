import os

from streamlit.web.bootstrap import run


def main() -> None:
    port = int(os.environ.get("PORT", "8501"))
    run(
        main_script_path="app/streamlit_app.py",
        is_hello=False,
        args=[],
        flag_options={
            "server.headless": True,
            "server.address": "0.0.0.0",
            "server.port": port,
            "browser.gatherUsageStats": False,
        },
    )


if __name__ == "__main__":
    main()
