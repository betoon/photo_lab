import numpy as np
from PyQt6.QtWidgets import QApplication

from restoration_dialog import RestorationStudioDialog


def test_apply_and_save_button_accepts_dialog_and_prepares_result(monkeypatch):
    app=QApplication.instance() or QApplication([])
    image=np.zeros((24,32,3),np.uint8);dialog=RestorationStudioDialog(None,image)
    expected=np.full_like(image,77);monkeypatch.setattr(dialog,"result",lambda:expected);accepted=[];dialog.accepted.connect(lambda:accepted.append(True))
    assert "Save Copy" in dialog.save_button.text()
    dialog.save_button.click();app.processEvents()
    assert accepted
    assert np.array_equal(dialog.result_image,expected)


def test_ai_tab_explains_temporary_candidates_and_offline_execution():
    app=QApplication.instance() or QApplication([])
    dialog=RestorationStudioDialog(None,np.zeros((24,32,3),np.uint8))
    assert "Offline" in dialog.run_ai_button.text()
    assert not dialog.export_candidate_button.isEnabled()
    assert "Save Copy" in dialog.status.text()
