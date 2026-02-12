"""Top-level Typer CLI wiring for the satellit command."""

import typer

from . import label as label_cli

app = typer.Typer(no_args_is_help=True)
app.add_typer(label_cli.app, name="label")


def main(argv: list[str] | None = None) -> None:
    """Execute the CLI application.

    Args:
        argv: Optional argument list for programmatic usage.
    """
    if argv is None:
        app()
        return
    app(args=argv, prog_name="satellit_sam")


if __name__ == "__main__":
    main()
