"""Qt GUI for editing constants and parameters in a SymbolicODE.

This module provides a table-based editor that allows users to:
- View all constants and parameters with their values and units
- Convert constants to parameters (swept) and vice versa via checkbox
- Edit values with forgiving float validation
"""

from typing import TYPE_CHECKING, Optional

from qtpy.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QCheckBox, QLineEdit, QPushButton, QLabel,
    QHeaderView, QMessageBox, QWidget,
)
from qtpy.QtCore import Qt

if TYPE_CHECKING:
    from cubie.odesystems.symbolic import SymbolicODE


class FloatLineEdit(QLineEdit):
    """Line edit with forgiving float validation.

    Accepts various numeric formats and provides visual feedback.
    """

    def __init__(self, value: float = 0.0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._value = value
        self.setText(self._format_value(value))
        self.editingFinished.connect(self._on_editing_finished)
        self._valid = True

    def _format_value(self, value: float) -> str:
        """Format value for display."""
        if abs(value) < 1e-4 or abs(value) >= 1e6:
            return f"{value:.6e}"
        return f"{value:.6g}"

    def _on_editing_finished(self) -> None:
        """Validate and update value when editing finishes."""
        text = self.text().strip()
        try:
            self._value = self._parse_float(text)
            self._valid = True
            self.setStyleSheet("")
            self.setText(self._format_value(self._value))
        except ValueError:
            self._valid = False
            self.setStyleSheet("background-color: #ffcccc;")

    def _parse_float(self, text: str) -> float:
        """Parse text to float with forgiving validation.

        Accepts:
        - Standard floats: 1.5, -2.3, .5
        - Scientific: 1e-5, 1.5E+10
        - Leading/trailing whitespace
        """
        text = text.strip().lower()
        if not text:
            return 0.0

        text = text.replace('d', 'e')

        return float(text)

    def value(self) -> float:
        """Return the current value."""
        return self._value

    def setValue(self, value: float) -> None:
        """Set the value."""
        self._value = value
        self.setText(self._format_value(value))
        self._valid = True
        self.setStyleSheet("")

    def isValid(self) -> bool:
        """Return whether the current text is valid."""
        return self._valid


class ConstantsEditor(QDialog):
    """Dialog for editing constants and parameters in a SymbolicODE.

    Parameters
    ----------
    ode
        The SymbolicODE instance to edit.
    parent
        Optional parent widget.

    Example
    -------
    >>> from cubie.odesystems.symbolic import load_cellml_model
    >>> ode = load_cellml_model("model.cellml")
    >>> editor = ConstantsEditor(ode)
    >>> editor.exec()  # Modal dialog
    """

    def __init__(
        self,
        ode: "SymbolicODE",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._ode = ode
        self._pending_changes: dict = {}
        self._checkboxes: dict[str, QCheckBox] = {}
        self._value_edits: dict[str, FloatLineEdit] = {}

        self._setup_ui()
        self._populate_table()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Constants & Parameters Editor")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Check 'Swept' to make a constant into a runtime parameter.\n"
            "Uncheck to convert a parameter back to a compile-time constant."
        )
        layout.addWidget(info_label)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Swept", "Value", "Unit"]
        )

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self._table.setColumnWidth(1, 60)
        self._table.setColumnWidth(2, 150)
        self._table.setColumnWidth(3, 100)

        layout.addWidget(self._table)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _populate_table(self) -> None:
        """Populate the table with constants and parameters."""
        constants = self._ode.get_constants_info()
        parameters = self._ode.get_parameters_info()

        all_items = []
        for info in constants:
            all_items.append({**info, 'is_parameter': False})
        for info in parameters:
            all_items.append({**info, 'is_parameter': True})

        all_items.sort(key=lambda x: x['name'])

        self._table.setRowCount(len(all_items))

        for row, item in enumerate(all_items):
            name = item['name']
            value = item['value']
            unit = item['unit']
            is_param = item['is_parameter']

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            checkbox = QCheckBox()
            checkbox.setChecked(is_param)
            checkbox.stateChanged.connect(
                lambda state, n=name: self._on_swept_changed(n, state)
            )
            self._checkboxes[name] = checkbox

            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(row, 1, checkbox_widget)

            value_edit = FloatLineEdit(value)
            value_edit.editingFinished.connect(
                lambda n=name: self._on_value_changed(n)
            )
            self._value_edits[name] = value_edit
            self._table.setCellWidget(row, 2, value_edit)

            unit_item = QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 3, unit_item)

    def _on_swept_changed(self, name: str, state: int) -> None:
        """Handle swept checkbox state change."""
        is_checked = state != 0
        self._pending_changes.setdefault(name, {})['swept'] = is_checked

    def _on_value_changed(self, name: str) -> None:
        """Handle value edit change."""
        edit = self._value_edits[name]
        if edit.isValid():
            self._pending_changes.setdefault(name, {})['value'] = edit.value()

    def _on_apply(self) -> None:
        """Apply all pending changes to the ODE."""
        invalid_edits = [
            name for name, edit in self._value_edits.items()
            if not edit.isValid()
        ]
        if invalid_edits:
            QMessageBox.warning(
                self,
                "Invalid Values",
                f"The following entries have invalid values: "
                f"{', '.join(invalid_edits)}"
            )
            return

        current_constants = set(self._ode.indices.constant_names)
        current_params = set(self._ode.indices.parameter_names)

        errors = []

        for name, changes in self._pending_changes.items():
            try:
                if 'swept' in changes:
                    if changes['swept'] and name in current_constants:
                        if 'value' in changes:
                            self._ode.set_constant_value(name, changes['value'])
                        self._ode.make_parameter(name)
                        current_constants.discard(name)
                        current_params.add(name)
                    elif not changes['swept'] and name in current_params:
                        if 'value' in changes:
                            self._ode.set_parameter_value(
                                name, changes['value']
                            )
                        self._ode.make_constant(name)
                        current_params.discard(name)
                        current_constants.add(name)
                    elif 'value' in changes:
                        if name in current_constants:
                            self._ode.set_constant_value(
                                name, changes['value']
                            )
                        else:
                            self._ode.set_parameter_value(
                                name, changes['value']
                            )
                elif 'value' in changes:
                    if name in current_constants:
                        self._ode.set_constant_value(name, changes['value'])
                    else:
                        self._ode.set_parameter_value(name, changes['value'])
            except Exception as e:
                errors.append(f"{name}: {e}")

        self._pending_changes.clear()

        if errors:
            QMessageBox.warning(
                self,
                "Errors Applying Changes",
                "Some changes could not be applied:\n" + "\n".join(errors)
            )
        else:
            QMessageBox.information(
                self,
                "Changes Applied",
                "All changes have been applied."
            )


def show_constants_editor(
    ode: "SymbolicODE",
    blocking: bool = True,
) -> Optional[ConstantsEditor]:
    """Show the constants/parameters editor dialog.

    Parameters
    ----------
    ode
        The SymbolicODE instance to edit.
    blocking
        If True, block until the dialog is closed. If False, return
        the dialog instance immediately.

    Returns
    -------
    ConstantsEditor or None
        The dialog instance if non-blocking, None if blocking.
    """
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    editor = ConstantsEditor(ode)

    if blocking:
        editor.exec() if hasattr(editor, 'exec') else editor.exec_()
        if created_app:
            app.quit()
        return None
    else:
        editor.show()
        return editor
