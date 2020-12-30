"""Clipster - Clipboard manager."""

# pylint: disable=line-too-long

from __future__ import print_function

import argparse as ap
from configparser import ConfigParser
from contextlib import closing
import errno
from getpass import getuser
import json
import logging
import os
import re
import signal
import socket
import stat
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from urllib.error import URLError

from gi import require_version
import prometheus_client as pc


require_version("Gdk", "3.0")
require_version("Gtk", "3.0")
from gi.repository import Gdk, GLib, GObject, Gtk


try:
    require_version("Wnck", "3.0")
    from gi.repository import Wnck
except (ImportError, ValueError):
    Wnck = None


logger = logging.getLogger(__name__)

AnyStr = Union[bytes, str]

PC_GATEWAY_HOST = "localhost:9091"
PC_HTTP_SERVER_PORT = 9102


class suppress_if_errno:
    """
    A context manager which suppresses exceptions with an errno attribute which
    matches the given value.

    Allows things like:

        try:
            os.makedirs(dirs)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    to be expressed as:

        with suppress_if_errno(OSError, errno.EEXIST):
            os.makedirs(dir)

    This is a fork of contextlib.suppress.

    """

    def __init__(
        self,
        exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
        exc_val: int,
    ) -> None:
        self._exceptions = exceptions
        self._exc_val = exc_val

    def __enter__(self) -> None:
        pass

    def __exit__(self, exctype: Any, excinst: Any, exctb: Any) -> Any:
        # Unlike isinstance and issubclass, CPython exception handling
        # currently only looks at the concrete type hierarchy (ignoring
        # the instance and subclass checking hooks). While Guido considers
        # that a bug rather than a feature, it's a fairly hard one to fix
        # due to various internal implementation details. suppress provides
        # the simpler issubclass based semantics, rather than trying to
        # exactly reproduce the limitations of the CPython interpreter.
        #
        # See http://bugs.python.org/issue12029 for more details
        return (
            exctype is not None
            and issubclass(exctype, self._exceptions)
            and excinst.errno == self._exc_val
        )


class ClipsterError(Exception):
    """Errors specific to Clipster."""

    def __init__(self, args: str = "Clipster Error.") -> None:
        Exception.__init__(self, args)


class Client:
    """Clipboard Manager."""

    def __init__(self, config: ConfigParser, args: ap.Namespace) -> None:
        self.config = config
        self.args = args
        self.client_action = "SEND"
        if args.select:
            self.client_action = "SELECT"
        elif args.ignore:
            self.client_action = "IGNORE"
        elif args.delete is not None:
            self.client_action = "DELETE"
        elif args.erase_entire_board:
            self.client_action = "ERASE"
        elif args.output or args.search is not None:
            self.client_action = "BOARD"
        logger.debug("client_action: %s", self.client_action)

    def update(self) -> None:
        """Send a signal and (optional) data from STDIN to daemon socket."""

        logger.debug("Connecting to server to update.")
        with closing(
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ) as sock:
            # pylint doesn't like contextlib.closing (https://github.com/PyCQA/astroid/issues/347)
            # pylint:disable=no-member
            try:
                sock.connect(self.config.get("clipster", "socket_file"))
            except (socket.error, OSError) as e:
                raise ClipsterError(
                    "Error connecting to socket. Is daemon running?"
                ) from e
            logger.debug("Sending request to server.")
            # Fix for http://bugs.python.org/issue1633941 in py 2.x
            # Send message 'header' - count is 0 (i.e to be ignored)
            sock.sendall(
                "{0}:{1}:0".format(
                    self.client_action,
                    self.config.get("clipster", "default_selection"),
                ).encode("utf-8")
            )

            if self.client_action == "DELETE":
                # Send delete args
                sock.sendall(":{0}".format(self.args.delete).encode("utf-8"))

            if self.client_action == "SEND":
                # Send data read from stdin
                buf_size = 8192
                # Send another colon to show that content is coming
                # Needed to distinguish empty content from no content
                # e.g. when stdin is empty
                sock.sendall(":".encode("utf-8"))
                while True:
                    if sys.stdin.isatty():
                        recv = sys.stdin.readline(buf_size)
                    else:
                        recv = sys.stdin.read(buf_size)
                    if not recv:
                        break

                    recv_str = safe_decode(recv)
                    assert isinstance(recv_str, str)
                    sock.sendall(recv_str.encode("utf-8"))

    def output(self) -> str:
        """Send a signal and count to daemon socket requesting items from history."""

        logger.debug("Connecting to server to query history.")
        with closing(
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ) as sock:
            # pylint doesn't like contextlib.closing (https://github.com/PyCQA/astroid/issues/347)
            # pylint:disable=no-member
            try:
                sock.connect(self.config.get("clipster", "socket_file"))
            except socket.error as e:
                raise ClipsterError(
                    "Error connecting to socket. Is daemon running?"
                ) from e
            logger.debug("Sending request to server.")
            # Send message 'header'
            message = "{0}:{1}:{2}".format(
                self.client_action,
                self.config.get("clipster", "default_selection"),
                self.args.number,
            )
            if self.args.search:
                message = "{0}:{1}".format(message, self.args.search)
            sock.sendall(message.encode("utf-8"))

            sock.shutdown(socket.SHUT_WR)
            data = []
            while True:
                try:
                    recv = sock.recv(8192)
                    logger.debug("Received data from server.")
                    if not recv:
                        break
                    data.append(safe_decode(recv))
                except socket.error:
                    break
        if data:
            # data is a list of 1 or more parts of a json string.
            # Reassemble this, then join with delimiter
            return self.args.delim.join(json.loads("".join(data)))  # type: ignore
        else:
            return ""


class Daemon:
    """Handles clipboard events, client requests, stores history."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, config: ConfigParser) -> None:
        """Set up clipboard objects and history dict."""

        self.config = config
        self.patterns: List[str] = []
        self.ignore_patterns: List[str] = []
        self.p_id: Optional[int] = None
        self.c_id: Optional[int] = None
        self.window: Optional[Gtk.Dialog] = None
        self.sock: Optional[socket.socket] = None
        self.sock_file = self.config.get("clipster", "socket_file")
        self.primary = Gtk.Clipboard.get(Gdk.SELECTION_PRIMARY)
        self.clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        self.boards: Dict[str, List[str]] = {"PRIMARY": [], "CLIPBOARD": []}
        self.hist_file = self.config.get("clipster", "history_file")
        self.pid_file = self.config.get("clipster", "pid_file")
        self.client_msgs: Dict[int, List[str]] = {}
        # Flag to indicate that the in-memory history should be flushed to disk
        self.update_history_file = False
        # Flag whether next clipboard change should be ignored
        self.ignore_next = {"PRIMARY": False, "CLIPBOARD": False}
        self.whitelist_classes = self.blacklist_classes = []
        if Wnck:
            self.blacklist_classes = get_list_from_option_string(
                self.config.get("clipster", "blacklist_classes")
            )
            self.whitelist_classes = get_list_from_option_string(
                self.config.get("clipster", "whitelist_classes")
            )
            if self.whitelist_classes:
                logger.debug(
                    "Whitelist classes enabled for: %s", self.whitelist_classes
                )
            if self.blacklist_classes:
                logger.debug(
                    "Blacklist classes enabled for: %s", self.blacklist_classes
                )
        else:
            logger.error(
                "'whitelist_classes' or 'blacklist_classes' require Wnck"
                " (libwnck3)."
            )

        self.pc_history_file_writes = pc.Counter(
            "clipster_history_file_writes",
            "Number of times the history file has been written to.",
        )

    def keypress_handler(
        self,
        _widget: Gtk.Widget,
        event: Gdk.Event,
        board: str,
        tree_select: Gtk.TreeSelection,
    ) -> None:
        """Handle selection_widget keypress events."""

        # Handle select with return or mouse
        if event.keyval == Gdk.KEY_Return:
            self.activate_handler(event, board, tree_select)
        # Delete items from history
        if event.keyval == Gdk.KEY_Delete:
            self.delete_handler(event, board, tree_select)
        # Hide window if ESC is pressed
        if event.keyval == Gdk.KEY_Escape:
            assert self.window is not None
            self.window.hide()

    def delete_handler(
        self, _event: Gdk.Event, board: str, tree_select: Gtk.TreeSelection
    ) -> None:
        """Delete selected history entries."""
        model, treepaths = tree_select.get_selected_rows()
        for tree in treepaths:
            treeiter = model.get_iter(tree)
            item = model[treeiter][1]
            item = safe_decode(item)
            logger.debug("Deleting history entry: %s", item)
            # If deleted item is currently on the clipboard, clear it
            if self.read_board(board) == item:
                self.update_board(board)
            # Remove item from history
            self.remove_history(board, item)
            if self.config.getboolean("clipster", "sync_selections"):
                # find the 'other' board
                board_list = list(self.boards)
                board_list.remove(board)
                # Is the other board active? If so, delete item from its history too
                if board_list[0] in self.config.get(
                    "clipster", "active_selections"
                ):
                    logger.debug("Synchronising delete to other board.")
                    # Remove item from history
                    self.remove_history(board_list[0], item)
                    # If deleted item is current on the clipboard, clear it
                    if self.read_board(board) == item:
                        self.update_board(board)
            # Remove entry from UI
            model.remove(treeiter)

    def activate_handler(
        self, _event: Gdk.Event, board: str, tree_select: Gtk.TreeSelection
    ) -> None:
        """Action selected history items."""

        # Get selection
        model, treepaths = tree_select.get_selected_rows()
        # Step over list in reverse, moving to top of board
        for tree in treepaths[::-1]:
            # Select full text from row
            data = model[model.get_iter(tree)][1]
            self.update_board(board, data)
            self.update_history(board, data)
        model.clear()
        assert self.window is not None
        self.window.hide()

    def selection_widget(self, board: str) -> None:
        """GUI window for selecting items from clipboard history."""

        # Create windows & widgets
        # Gtk complains about dialogs with no parents, so create one
        self.window = Gtk.Dialog(title="Clipster", parent=Gtk.Window())
        scrolled = Gtk.ScrolledWindow()
        model = Gtk.ListStore(str, str)
        tree = Gtk.TreeView(model)
        tree_select = tree.get_selection()
        tree_select.set_mode(Gtk.SelectionMode.MULTIPLE)
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn(
            "{0} clipboard:\n <ret> to activate, <del> to remove, <esc> to"
            " exit.".format(board),
            renderer,
            markup=0,
        )

        # Add rows to the model
        for item in self.boards[board][::-1]:
            label = GLib.markup_escape_text(item)
            row_height = self.config.getint("clipster", "row_height")
            trunc = ""
            lines = label.splitlines(True)
            if len(lines) > row_height + 1:
                trunc = "<b><i>({0} more lines)</i></b>".format(
                    len(lines) - row_height
                )
            label = "{0}{1}".format("".join(lines[:row_height]), trunc)
            # Add label and full text to model
            model.append([label, item])

        # Format, connect and show windows
        # Allow alternating color for rows, if WM theme supports it
        tree.set_rules_hint(True)
        # Draw horizontal divider lines between rows
        tree.set_grid_lines(Gtk.TreeViewGridLines.HORIZONTAL)

        tree.append_column(column)
        scrolled.add(tree)

        # Handle keypresses
        self.window.connect(
            "key-press-event", self.keypress_handler, board, tree_select
        )

        # Handle window delete event
        self.window.connect("delete_event", self.window.hide)

        # Add a 'select' button
        select_btn = Gtk.Button.new_with_label("Select")
        select_btn.connect(
            "clicked", self.activate_handler, board, tree_select
        )

        # Add a box to hold buttons
        button_box = Gtk.Box()
        button_box.pack_start(select_btn, True, False, 0)

        # GtkDialog comes with a vbox already active, so pack into this
        self.window.vbox.pack_start(
            scrolled, True, True, 0
        )  # pylint: disable=no-member
        self.window.vbox.pack_start(
            button_box, False, False, 0
        )  # pylint: disable=no-member
        self.window.set_size_request(500, 500)
        self.window.show_all()

    def read_history_file(self) -> None:
        """Read clipboard history from file."""

        try:
            with open(self.hist_file) as hist_f:
                self.boards.update(json.load(hist_f))
        except FileNotFoundError as exc:
            if exc.errno != errno.ENOENT:
                # Not an error if there is no history file
                raise

    def write_history_file(self) -> bool:
        """Write clipboard history to file."""

        if self.update_history_file:
            self.pc_history_file_writes.inc()

            # Limit history file to contain last 'history_size' items
            limit = self.config.getint("clipster", "history_size")
            # If limit is 0, don't write to file
            if limit:
                hist = {x: y[-limit:] for x, y in self.boards.items()}
                logger.debug("Writing history to file.")
                with tempfile.NamedTemporaryFile(
                    dir=self.config.get("clipster", "data_dir"), delete=False
                ) as tmp_file:
                    tmp_file.write(json.dumps(hist).encode("utf-8"))
                os.rename(tmp_file.name, self.hist_file)
                self.update_history_file = False
        else:
            logger.debug("History unchanged - not writing to file.")
        # Return true to make the timeout handler recur
        return True

    def read_board(self, board: str) -> AnyStr:
        """Return the text on the clipboard."""

        return safe_decode(getattr(self, board.lower()).wait_for_text())

    def update_board(self, board: str, data: str = "") -> None:
        """Update a clipboard. Will trigger an owner-change event."""

        selection = getattr(self, board.lower())
        selection.set_text(data, -1)
        if not data:
            selection.clear()

    def remove_history(self, board: str, text: str) -> None:
        """If text exists in the history, remove it."""

        if text in self.boards[board]:
            logger.debug("Removing from history.")
            self.boards[board].remove(text)
            # Flag the history file for updating
            self.update_history_file = True

    def update_history(self, board: str, text: str) -> None:  # noqa: C901
        """Update the in-memory clipboard history."""

        for ignore in self.ignore_patterns:
            # If text matches an ignore pattern, don't update history
            if re.search(ignore, text):
                logger.debug(
                    "Pattern: '%s' matches selection: '%s' - ignoring.",
                    ignore,
                    text,
                )
                return

        if self.ignore_next[board]:
            # Ignore history update this time and reset ignore flag
            logger.debug("Ignoring update of %s history", board)
            self.ignore_next[board] = False
            return

        logger.debug("Updating clipboard: %s", board)

        text_str = safe_decode(text)
        assert isinstance(text_str, str)

        if not self.config.getboolean("clipster", "duplicates"):
            self.remove_history(board, text_str)
        diff = self.config.getint("clipster", "smart_update")
        try:
            last_item = self.boards[board][-1]
        except IndexError:
            # List was empty
            last_item = ""
        # Check for growing or shrinking, but ignore duplicates
        if (
            last_item
            and text != last_item
            and (text in last_item or last_item in text)
        ):
            # Make length difference a positive number before comparing
            if abs(len(text) - len(last_item)) <= diff:
                logger.debug("smart-update: removing.")
                # new selection is a longer/shorter version of previous
                self.boards[board].pop()

        if self.config.getboolean("clipster", "extract_uris"):
            # Simple uri regex
            self.patterns.insert(0, r"\b\S+://\S+\b")
        if self.config.getboolean("clipster", "extract_emails"):
            # Simple email regex
            self.patterns.insert(0, r"\b\S+\@\S+\.\S+\b")

        # Insert selection into history before pattern matching
        self.boards[board].append(text)

        for pattern in self.patterns:
            try:
                for match in set(re.findall(pattern, text)):
                    if match != text:
                        logger.debug(
                            "Pattern '%s' matched in: %s", pattern, text
                        )
                        if not self.config.getboolean(
                            "clipster", "duplicates"
                        ):
                            self.remove_history(board, match)
                        if self.config.getboolean(
                            "clipster", "pattern_as_selection"
                        ):
                            self.ignore_next[board] = True
                            self.update_board(board, match)
                            self.boards[board].append(match)
                        else:
                            self.boards[board].insert(-1, match)
            except re.error as exc:
                logger.warning(
                    "Skipping invalid pattern '%s': %s", pattern, exc.args[0]
                )

        # Flag that the history file needs updating
        self.update_history_file = True
        if self.config.getboolean("clipster", "write_on_change"):
            self.write_history_file()
        logger.debug(self.boards[board])
        if self.config.getboolean("clipster", "sync_selections"):
            # Whichever board we just set, set the other one, if it's active
            boards = list(self.boards)
            boards.remove(board)
            # Stop if the board already contains the text.
            if (
                boards[0] in self.config.get("clipster", "active_selections")
                and self.read_board(boards[0]) != text
            ):
                logger.debug("Syncing board %s to %s", board, boards[0])
                self.update_board(boards[0], text)

    def owner_change(self, board: Gtk.Clipboard, event: Gdk.Event) -> bool:
        """Handler for owner-change clipboard events."""

        logger.debug("owner-change event!")
        selection = str(event.selection)
        logger.debug("selection: %s", selection)
        active = self.config.get("clipster", "active_selections").split(",")

        if selection not in active:
            return False

        # Only monitor owner-change events for apps with WM_CLASS values found
        # in whitelist and not found in blacklist
        if self.whitelist_classes or self.blacklist_classes:
            wm_class = get_wm_class_from_active_window().lower()
            if (
                self.whitelist_classes
                and wm_class not in self.whitelist_classes
            ) or (
                self.blacklist_classes and wm_class in self.blacklist_classes
            ):
                logger.debug("Ignoring active window.")
                return True

        logger.debug("Selection in 'active_selections'")
        event_id = self.p_id if selection == "PRIMARY" else self.c_id
        # Some apps update primary during mouse drag (chrome)
        # Block at start to prevent repeated triggering
        board.handler_block(event_id)

        assert self.window is not None
        display = self.window.get_display()

        while Gdk.ModifierType.BUTTON1_MASK & display.get_pointer().mask:
            # Do nothing while mouse button is held down (selection drag)
            pass
        # Try to get text from clipboard
        text = board.wait_for_text()
        if text:
            logger.debug("Selection is text.")
            self.update_history(selection, text)
            # If no text received, either the selection was an empty string,
            # or the board contains non-text content.
        else:
            # First item in tuple is bool, False if no targets
            if board.wait_for_targets()[0]:
                logger.debug("Selection is not text - ignoring.")
            else:
                logger.debug(
                    "Clipboard cleared or empty. Reinstating from history."
                )
                if self.boards[selection]:
                    self.update_board(selection, self.boards[selection][-1])
                else:
                    logger.debug(
                        "No history available, leaving clipboard empty."
                    )

        # Unblock event handling
        board.handler_unblock(event_id)
        return False

    def socket_accept(self, sock: socket.socket, _: Any) -> bool:
        """Accept a connection and 'select' it for readability."""

        conn, _ = sock.accept()
        self.client_msgs[conn.fileno()] = []
        GObject.io_add_watch(conn, GObject.IO_IN, self.socket_recv)
        logger.debug("Client connection received.")
        return True

    def socket_recv(self, conn: socket.socket, _: Any) -> bool:
        """Try to recv from an accepted connection."""

        max_input = self.config.getint("clipster", "max_input")
        recv_total = sum(len(x) for x in self.client_msgs[conn.fileno()])
        try:
            recv = safe_decode(conn.recv(min(8192, max_input - recv_total)))
            assert isinstance(recv, str)
            self.client_msgs[conn.fileno()].append(recv)
            recv_total += len(recv)
            if not recv or recv_total >= max_input:
                self.process_msg(conn)
            else:
                return True
        except socket.error as exc:
            logger.error("Socket error %s", exc)
            logger.debug("Exception:", exc_info=True)

        conn.close()
        # Return false to remove conn from GObject.io_add_watch list
        return False

    def process_msg(self, conn: socket.socket) -> None:  # noqa: C901
        """Process message received from client, sending reply if required."""

        try:
            msg_str = "".join(self.client_msgs.pop(conn.fileno()))
        except KeyError:
            return
        try:
            msg_parts = msg_str.split(":", 3)
            content: Optional[str]
            if len(msg_parts) == 4:
                sig, board, count_str, content = msg_parts
            elif len(msg_parts) == 3:
                sig, board, count_str = msg_parts
                content = None
            else:
                raise ValueError()
            count = int(count_str)
        except (TypeError, ValueError):
            logger.error("Invalid message received via socket: %s", msg_str)
            return
        logger.debug("Received: sig:%s, board:%s, count:%s", sig, board, count)
        if sig == "SELECT":
            self.selection_widget(board)
        elif sig == "SEND":
            if content is not None:
                logger.debug("Received content: %s", content)
                self.update_board(board, content)
            else:
                raise ClipsterError("No content received!")
        elif sig == "BOARD":
            result = self.boards[board]
            if content:
                logger.debug("Searching for pattern: %s", content)
                result = [
                    x for x in self.boards[board] if re.search(content, x)
                ]
            logger.debug(
                "Sending requested selection(s): %s", result[-count:][::-1]
            )
            # Send list (reversed) as json to preserve structure
            try:
                conn.sendall(json.dumps(result[-count:][::-1]).encode("utf-8"))
            except (socket.error, OSError) as exc:
                logger.error("Socket error %s", exc)
                logger.debug("Exception:", exc_info=True)
        elif sig == "IGNORE":
            self.ignore_next[board] = True
        elif sig == "DELETE":
            if content:
                logger.debug(
                    "Deleting clipboard items matching text: %s", content
                )
                self.remove_history(board, content)
                # If deleted item is current on the clipboard, clear it
                if self.read_board(board) == content:
                    self.update_board(board)
            else:
                try:
                    logger.debug("Deleting last item in history.")
                    last = self.boards[board].pop()
                    # If deleted item is current on the clipboard, clear it
                    if self.read_board(board) == last:
                        self.update_board(board)
                except IndexError:
                    logger.debug("History already empty.")
        elif sig == "ERASE":
            logger.debug(
                "Erasing clipboard (%d items)", len(self.boards[board])
            )
            self.boards[board] = []
            self.update_board(board)
            self.update_history_file = True

    def read_patt_file(self, name: str) -> List[str]:
        """Get a series of regexes (one per line) from a file and return as a list."""
        try:
            patfile = os.path.join(
                self.config.get("clipster", "conf_dir"), name
            )
            with open(patfile) as pat_f:
                patts = [x.strip() for x in pat_f.read().splitlines()]
                logger.debug("Loaded patterns: %s", ",".join(patts))
                return patts
        except FileNotFoundError as exc:
            logger.warning(
                "Unable to read patterns file: %s %s", patfile, exc.strerror
            )
            return []

    def prepare_files(self) -> None:
        """Ensure that all files and sockets used
        by the daemon are available."""

        # Create the clipster dir if necessary
        with suppress_if_errno(FileExistsError, errno.EEXIST):
            os.makedirs(self.config.get("clipster", "data_dir"))

        # check for existing pid_file, and tidy up if appropriate
        with suppress_if_errno(FileNotFoundError, errno.ENOENT):
            with open(self.pid_file) as runf_r:
                try:
                    pid = int(runf_r.read())
                except ValueError:
                    logger.debug("Invalid pid file, attempting to overwrite.")
                else:
                    # pid is an int, determine if this corresponds to a running daemon.
                    try:
                        # Do nothing, but raise an error if no such process
                        os.kill(pid, 0)
                        raise ClipsterError(
                            "Daemon already running: pid {}".format(pid)
                        )
                    except ProcessLookupError as exc:
                        if exc.errno != errno.ESRCH:
                            raise
                        # No process found, delete the pid file.
                        with suppress_if_errno(
                            FileNotFoundError, errno.ENOENT
                        ):
                            os.unlink(self.pid_file)

        # Create pid file
        with open(self.pid_file, "w") as runf_w:
            runf_w.write(str(os.getpid()))

        # Read in history from file
        self.read_history_file()

        # Create the socket
        with suppress_if_errno(FileNotFoundError, errno.ENOENT):
            os.unlink(self.sock_file)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.sock_file)
        os.chmod(self.sock_file, stat.S_IRUSR | stat.S_IWUSR)
        self.sock.listen(5)

        # Read in patterns file
        if self.config.getboolean("clipster", "extract_patterns"):
            logger.debug("extract_patterns enabled.")
            self.patterns = self.read_patt_file(
                self.config.get("clipster", "extract_patterns_file")
            )

        # Read in ignore_patterns file
        if self.config.getboolean("clipster", "ignore_patterns"):
            logger.debug("ignore_patterns enabled.")
            self.ignore_patterns = self.read_patt_file(
                self.config.get("clipster", "ignore_patterns_file")
            )

    def exit(self) -> None:
        """Clean up things before exiting."""

        logger.debug("Daemon exiting...")
        try:
            os.unlink(self.sock_file)
        except FileNotFoundError:
            logger.warning("Failed to remove socket file: %s", self.sock_file)
        try:
            os.unlink(self.pid_file)
        except FileNotFoundError:
            logger.warning("Failed to remove pid file: %s", self.pid_file)
        try:
            self.write_history_file()
        except FileNotFoundError:
            logger.warning("Failed to update history file: %s", self.hist_file)
        Gtk.main_quit()

    def run(self) -> None:
        """Launch the clipboard manager daemon.
        Listen for clipboard events & client socket connections."""

        # Set up socket, pid file etc
        self.prepare_files()

        # Start the prometheus metrics client.
        try:
            pc.start_http_server(PC_HTTP_SERVER_PORT)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                logger.warning(
                    "The prometheus server failed to start, since the network"
                    " port %d is already in use.",
                    PC_HTTP_SERVER_PORT,
                )
            else:
                raise

        # We need to get the display instance from the window
        # for use in obtaining mouse state.
        # POPUP windows can do this without having to first show the window
        self.window = Gtk.Window(type=Gtk.WindowType.POPUP)

        # Handle clipboard changes
        self.p_id = self.primary.connect("owner-change", self.owner_change)
        self.c_id = self.clipboard.connect("owner-change", self.owner_change)
        # Handle socket connections
        GObject.io_add_watch(self.sock, GObject.IO_IN, self.socket_accept)
        # Handle unix signals
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, self.exit)
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGTERM, self.exit)
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGHUP, self.exit)

        # Timeout for flushing history to disk
        # Do nothing if timeout is 0, or write_on_change is set in config
        history_timeout = self.config.getint(
            "clipster", "history_update_interval"
        )
        if history_timeout and not self.config.getboolean(
            "clipster", "write_on_change"
        ):
            logger.debug(
                "Writing history file every %s seconds", history_timeout
            )
            GObject.timeout_add_seconds(
                history_timeout, self.write_history_file
            )

        Gtk.main()


def get_wm_class_from_active_window() -> str:
    """Returns the current active window's WM_CLASS"""
    screen = Wnck.Screen.get_default()
    screen.force_update()
    active_window = screen.get_active_window()
    wm_class = active_window.get_class_group_name()
    logger.debug("Active window class is %s", wm_class)
    return wm_class


def get_list_from_option_string(string: str) -> List[str]:
    """Parse a configured option's string of elements,
    splits it around "," and returns a list of items in lower case,
    or an empty list if string was empty."""
    if string and string != r'""':
        return string.lower().split(",")
    return []


def parse_args() -> ap.Namespace:
    """Parse command-line arguments."""

    parser = ap.ArgumentParser(description="Clipster clipboard manager.")
    parser.add_argument(
        "-f", "--config", action="store", help="Path to config directory."
    )
    parser.add_argument(
        "-l",
        "--log_level",
        action="store",
        default="INFO",
        help="Set log level: DEBUG, INFO (default), WARNING, ERROR, CRITICAL",
    )
    # Mutually exclusive client and daemon options.
    boardgrp = parser.add_mutually_exclusive_group()
    boardgrp.add_argument(
        "-p",
        "--primary",
        action="store_const",
        const="PRIMARY",
        help="Query, or write STDIN to, the PRIMARY clipboard.",
    )
    boardgrp.add_argument(
        "-c",
        "--clipboard",
        action="store_const",
        const="CLIPBOARD",
        help="Query, or write STDIN to, the CLIPBOARD clipboard.",
    )
    boardgrp.add_argument(
        "-d", "--daemon", action="store_true", help="Launch the daemon."
    )

    # Mutually exclusive client actions
    actiongrp = parser.add_mutually_exclusive_group()
    actiongrp.add_argument(
        "-s",
        "--select",
        action="store_true",
        help="Launch the clipboard history selection window.",
    )
    actiongrp.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="Output selection from history. (See -n and -S).",
    )
    actiongrp.add_argument(
        "-i",
        "--ignore",
        action="store_true",
        help="Instruct daemon to ignore next update to clipboard.",
    )
    actiongrp.add_argument(
        "-r",
        "--delete",
        action="store",
        nargs="?",
        const="",
        help=(
            "Delete from clipboard. Deletes matching text, or if no argument"
            " given, deletes last item."
        ),
    )
    actiongrp.add_argument(
        "--erase-entire-board",
        action="store_true",
        help="Delete all items from the clipboard.",
    )

    parser.add_argument(
        "-n",
        "--number",
        action="store",
        type=int,
        default=1,
        help=(
            "Number of lines to output: defaults to 1 (See -o). 0 returns"
            " entire history."
        ),
    )

    parser.add_argument(
        "-S", "--search", action="store", help="Pattern to match for output."
    )

    # --delim must come before -0 to ensure delim is set correctly
    # otherwise if neither arg is passed, delim=None
    parser.add_argument(
        "-m",
        "--delim",
        action="store",
        default="\n",
        help="String to use as output delimiter (defaults to '\n')",
    )
    parser.add_argument(
        "-0",
        "--nul",
        action="store_const",
        const="\0",
        dest="delim",
        help="Use NUL character as output delimiter.",
    )

    return parser.parse_args()


def parse_config(
    args: ap.Namespace, data_dir: str, conf_dir: str
) -> ConfigParser:
    """Configuration derived from defaults & file."""

    # Set some config defaults
    config_defaults = {
        "data_dir": data_dir,  # clipster 'root' dir (see history/socket config)
        "conf_dir": conf_dir,  # clipster config dir (see pattern/ignore_pattern file config). Can be overridden using -f cmd-line arg.
        "default_selection": "PRIMARY",  # PRIMARY or CLIPBOARD
        "active_selections": "PRIMARY,CLIPBOARD",  # Comma-separated list of selections to monitor/save
        "sync_selections": "no",  # Synchronise contents of both clipboards
        "history_file": "%(data_dir)s/history",
        "history_size": "200",  # Number of items to be saved in the history file (for each selection)
        "history_update_interval": "60",  # Flush history to disk every N seconds, if changed (0 disables timeout)
        "write_on_change": "no",  # Always write history file immediately (overrides history_update_interval)
        "socket_file": "%(data_dir)s/clipster_sock",
        "pid_file": "/run/user/{}/clipster.pid".format(os.getuid()),
        "max_input": "50000",  # max length of selection input
        "row_height": "3",  # num rows to show in widget
        "duplicates": "no",  # allow duplicates, or instead move the original entry to top
        "smart_update": "1",  # Replace rather than append if selection is similar to previous
        "extract_uris": "yes",  # Extract URIs within selection text
        "extract_emails": "yes",  # Extract emails within selection text
        "extract_patterns": "no",  # Extract patterns based on regexes stored in data_dir/patterns (one per line).
        "extract_patterns_file": "%(conf_dir)s/patterns",  # patterns file for extract_patterns
        "ignore_patterns": "no",  # Ignore selections which match regex patterns stored in data_dir/ignore_patterns (one per line).
        "ignore_patterns_file": "%(conf_dir)s/ignore_patterns",  # patterns file for ignore_patterns
        "pattern_as_selection": "no",  # Extracted pattern should replace current selection.
        "blacklist_classes": "",  # Comma-separated list of WM_CLASS to identify apps from which to ignore owner-change events
        "whitelist_classes": "",
    }  # Comma-separated list of WM_CLASS to identify apps from which to not ignore owner-change events

    config = ConfigParser(config_defaults)
    config.add_section("clipster")

    # Try to read config file (either passed in, or default value)
    if args.config:
        config.set("clipster", "conf_dir", args.config)
    conf_file = os.path.join(
        config.get("clipster", "conf_dir"), "clipster.ini"
    )
    logger.debug("Trying to read config file: %s", conf_file)
    result = config.read(conf_file)
    if not result:
        logger.debug("Unable to read config file: %s", conf_file)

    logger.debug(
        "Merged config: %s", sorted(dict(config.items("clipster")).items())
    )

    return config


def find_config() -> Tuple[str, str]:
    """Attempt to find config from xdg basedir-spec paths/environment variables."""

    # Set a default directory for clipster files
    # https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
    xdg_config_dirs = os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")
    xdg_config_dirs.insert(
        0,
        os.environ.get(
            "XDG_CONFIG_HOME", os.path.join(os.environ["HOME"], ".config")
        ),
    )
    xdg_data_home = os.environ.get(
        "XDG_DATA_HOME", os.path.join(os.environ["HOME"], ".local/share")
    )

    data_dir = os.path.join(xdg_data_home, "clipster")
    # Keep trying to define conf_dir, moving from local -> global
    for path in xdg_config_dirs:
        conf_dir = os.path.join(path, "clipster")
        if os.path.exists(conf_dir):
            return conf_dir, data_dir
    return "", data_dir


def safe_decode(data: AnyStr) -> AnyStr:
    """Convenience method to ensure everything is utf-8."""

    try:
        result = data.decode("utf-8")  # type: ignore
    except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
        result = data

    return result


def configure_logging(log_level_str: str) -> None:
    log_level = getattr(logging, log_level_str.upper())

    fmt = (
        "%(asctime)s.%(msecs)-3d  |  %(levelname)-7s  |  PID:%(process)d  | "
        " %(message)s  [%(module)s::%(funcName)s::%(lineno)d]"
    )
    formatter = lambda datefmt: logging.Formatter(fmt, datefmt=datefmt)

    root = logging.root
    root.setLevel(log_level)

    sh = logging.StreamHandler()
    sh.formatter = formatter("%H:%M:%S")
    root.addHandler(sh)

    log_basename = "clipster.log"
    log_fname: Optional[str] = None
    all_log_dirs = [
        "/var/log",
        "/var/tmp",
        "/tmp",
        os.environ.get("HOME", f"/home/{getuser()}"),
    ]
    for log_dirname in all_log_dirs:
        if os.access(log_dirname, os.W_OK):
            log_fname = f"{log_dirname}/{log_basename}"
            break

    if log_fname is None:
        logger.warning(
            "We do not have permission to write to any of the following"
            f" directories and thus cannot create a log file: {all_log_dirs}"
        )
    else:
        fh = logging.FileHandler(log_fname)
        fh.formatter = formatter("%Y-%m-%d %H:%M:%S")
        root.addHandler(fh)

        logger.debug("Logging to %s.", log_fname)

    logger.debug("Debugging Enabled.")


def main() -> int:
    """Start the application. Return an exit status (0 or 1)."""

    try:
        # Find default config and data dirs
        conf_dir, data_dir = find_config()

        # parse command-line arguments
        args = parse_args()

        # Enable logging
        configure_logging(args.log_level.upper())

        config = parse_config(args, data_dir, conf_dir)

        # Launch the daemon
        if args.daemon:
            Daemon(config).run()
        else:
            board = (
                args.primary
                or args.clipboard
                or config.get("clipster", "default_selection")
            )
            if board not in config.get("clipster", "active_selections"):
                raise ClipsterError(
                    "{0} not in 'active_selections' in config.".format(board)
                )
            config.set("clipster", "default_selection", board)
            client = Client(config, args)

            pc_registry = pc.CollectorRegistry()
            pc_history_count = pc.Counter(
                "clipster_history_count",
                "Count of items retrieved from clipster's history file by"
                " client.",
                registry=pc_registry,
            )
            if args.output:
                # Ask server for clipboard history
                output = client.output()
                pc_history_count.inc(len(output.split(args.delim)))
                print(output, end="")
            else:
                # Read from stdin and send to server
                client.update()

            try:
                pc.pushadd_to_gateway(
                    PC_GATEWAY_HOST,
                    job="clipster_client",
                    registry=pc_registry,
                )
            except URLError:
                logger.warning(
                    "The prometheus PushGateway does not seem to be online"
                    " (%s).",
                    PC_GATEWAY_HOST,
                )
    except ClipsterError as exc:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            raise

        # Only output the 'human-readable' part.
        logger.error(exc)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
