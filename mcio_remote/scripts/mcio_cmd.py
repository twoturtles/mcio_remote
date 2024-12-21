import argparse
import pprint
import typing
from typing import Any
from pathlib import Path
import textwrap

from mcio_remote import instance
from mcio_remote import config
from mcio_remote import world
from mcio_remote import mcio_gui


def _add_mcio_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mcio-dir",
        "-d",
        type=str,
        default=config.DEFAULT_MCIO_DIR,
        help=f"MCio data directory (default: {config.DEFAULT_MCIO_DIR})",
    )


class ShowCmd:
    CMD = "show"

    def cmd(self) -> str:
        return self.CMD

    # Unfortunately, argparse is not set up for type hints
    def add(self, parent_subparsers: "argparse._SubParsersAction[Any]") -> None:
        show_parser = parent_subparsers.add_parser(
            self.CMD, help="Show information about what is installed"
        )
        _add_mcio_dir_arg(show_parser)

    def run(self, args: argparse.Namespace) -> None:
        self.show(args.mcio_dir)

    def show(self, mcio_dir: Path | str) -> None:
        mcio_dir = Path(mcio_dir).expanduser()
        print(f"Showing information for MCio directory: {mcio_dir}")
        with config.ConfigManager(mcio_dir) as cm:
            print("\nInstances:")
            for inst_id, inst_cfg in cm.config.instances.items():
                print(f"  {inst_id}: mc_version={inst_cfg.minecraft_version}")
                saves_dir = instance.get_saves_dir(mcio_dir, inst_id)
                if saves_dir.exists():
                    print("    Worlds:")
                    for world_path in saves_dir.iterdir():
                        print(f"      {world_path.name}")

            print("\nWorld Storage:")
            for world_name, world_cfg in cm.config.world_storage.items():
                print(
                    f"  {world_name}: mc_version={world_cfg.minecraft_version} seed={world_cfg.seed}"
                )

            print()


class WorldCmd:
    CMD = "world"

    def cmd(self) -> str:
        return self.CMD

    def run(self, args: argparse.Namespace) -> None:
        if args.world_command == "cp":
            wrld = world.World(mcio_dir=args.mcio_dir)
            wrld.copy(args.src, args.dst)
        elif args.world_command == "create":
            wrld = world.World(mcio_dir=args.mcio_dir)
            wrld.create(args.world_name, args.version, seed=args.seed)

    def add(self, parent_subparsers: "argparse._SubParsersAction[Any]") -> None:
        """Add the world command subparser"""
        world_parser = parent_subparsers.add_parser("world", help="World management")
        world_subparsers = world_parser.add_subparsers(
            dest="world_command", metavar="world-command", required=True
        )

        create_parser = world_subparsers.add_parser("create", help="Create a new world")
        create_parser.add_argument(
            "world_name",
            metavar="world-name",
            type=str,
            help="Name of the world",
        )
        _add_mcio_dir_arg(create_parser)
        create_parser.add_argument(
            "--version",
            "-v",
            type=str,
            default=config.DEFAULT_MINECRAFT_VERSION,
            help=f"World's Minecraft version (default: {config.DEFAULT_MINECRAFT_VERSION})",
        )
        create_parser.add_argument(
            "--seed",
            "-s",
            type=str,
            help="Set the world's seed (default is a random seed)",
        )

        cp_parser = world_subparsers.add_parser("cp", help="Copy a world")
        _add_mcio_dir_arg(cp_parser)
        cp_parser.add_argument(
            "src",
            type=str,
            help="Source world (storage:<world-name> or <instance-name>:<world-name>)",
        )
        cp_parser.add_argument(
            "dst",
            type=str,
            help="Dest world (storage:<world-name> or <instance-name>:<world-name>)",
        )


class GuiCmd:
    CMD = "gui"

    def cmd(self) -> str:
        return self.CMD

    def run(self, args: argparse.Namespace) -> None:
        gui = mcio_gui.MCioGUI(scale=args.scale, fps=args.fps)
        gui.run()

    def add(self, parent_subparsers: "argparse._SubParsersAction[Any]") -> None:
        """Add the gui command subparser"""
        gui_parser = parent_subparsers.add_parser(
            "gui",
            help="Launch demo GUI",
            description=textwrap.dedent(
                """
                Provides a human GUI to MCio.
                Q to quit.
                                        """
            ),
        )
        gui_parser.add_argument(
            "--scale",
            type=float,
            default=1.0,
            help="Window scale factor",
        )
        gui_parser.add_argument("--fps", type=int, default=60, help="Set fps limit")


class LaunchCmd:
    CMD = "launch"

    def cmd(self) -> str:
        return self.CMD

    def run(self, args: argparse.Namespace) -> None:
        launch = instance.Launcher(
            args.instance_id,
            mcio_dir=args.mcio_dir,
            mc_username=args.username,
            world_name=args.world,
            width=args.width,
            height=args.height,
            mcio_mode=args.mcio_mode,
        )
        if args.list:
            cmd = launch.get_show_command()
            pprint.pprint(cmd)
        elif args.str:
            cmd = launch.get_show_command()
            print(" ".join(cmd))
        else:
            launch.launch()

    def add(self, parent_subparsers: "argparse._SubParsersAction[Any]") -> None:
        launch_parser = parent_subparsers.add_parser("launch", help="Launch Minecraft")
        launch_parser.add_argument(
            "instance_id",
            metavar="instance-id",
            type=str,
            help="ID/Name of the Minecraft instance",
        )
        launch_parser.add_argument(
            "--mcio_mode",
            "-m",
            metavar="mcio-mode",
            type=str,
            choices=typing.get_args(instance.McioMode),
            default="async",
            help="MCio mode: (default: async)",
        )
        _add_mcio_dir_arg(launch_parser)
        launch_parser.add_argument("--world", "-w", type=str, help="World name")
        launch_parser.add_argument(
            "--width",
            "-W",
            type=int,
            default=instance.DEFAULT_WINDOW_WIDTH,
            help=f"Window width (default: {instance.DEFAULT_WINDOW_WIDTH})",
        )
        launch_parser.add_argument(
            "--height",
            "-H",
            type=int,
            default=instance.DEFAULT_WINDOW_HEIGHT,
            help=f"Window height (default: {instance.DEFAULT_WINDOW_HEIGHT})",
        )
        launch_parser.add_argument(
            "--username",
            "-u",
            type=str,
            default=instance.DEFAULT_MINECRAFT_USER,
            help=f"Player name (default: {instance.DEFAULT_MINECRAFT_USER})",
        )

        launch_group = launch_parser.add_mutually_exclusive_group()
        launch_group.add_argument(
            "--list",
            action="store_true",
            default=False,
            help="Don't run the command; print it as a list",
        )
        launch_group.add_argument(
            "--str",
            action="store_true",
            default=False,
            help="Don't run the command; print it as a string",
        )


class InstallCmd:
    CMD = "install"

    def run(self, args: argparse.Namespace) -> None:
        installer = instance.Installer(args.instance_id, args.mcio_dir, args.version)
        installer.install()

    def add(self, parent_subparsers: "argparse._SubParsersAction[Any]") -> None:
        install_parser = parent_subparsers.add_parser(
            "install", help="Install Minecraft"
        )
        install_parser.add_argument(
            "instance_id",
            metavar="instance-id",
            type=str,
            help="ID/Name of the Minecraft instance",
        )
        _add_mcio_dir_arg(install_parser)
        install_parser.add_argument(
            "--version",
            "-v",
            type=str,
            default=config.DEFAULT_MINECRAFT_VERSION,
            help=f"Minecraft version to install (default: {config.DEFAULT_MINECRAFT_VERSION})",
        )


def parse_args() -> tuple[argparse.Namespace, list[Any]]:
    parser = argparse.ArgumentParser(
        description="Minecraft Instance Manager and Launcher"
    )

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    cmd_objects: list[Any] = [
        InstallCmd(),
        LaunchCmd(),
        WorldCmd(),
        GuiCmd(),
        ShowCmd(),
    ]

    for cmd in cmd_objects:
        cmd.add(subparsers)

    return parser.parse_args(), cmd_objects


def main() -> None:
    args, cmd_objects = parse_args()
    for cmd in cmd_objects:
        if args.command == cmd.cmd():
            cmd.run(args)
            return
    print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
