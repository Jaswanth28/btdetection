/*
This is a UI file (.ui.qml) that is intended to be edited in Qt Design Studio only.
It is supposed to be strictly declarative and only uses a subset of QML. If you edit
this file manually, you might introduce QML code that is not supported by Qt Design Studio.
Check out https://doc.qt.io/qtcreator/creator-quick-ui-forms.html for details on .ui.qml files.
*/

import QtQuick 6.5
import QtQuick.Controls 6.5
import BTUI

Rectangle {

    color: Constants.backgroundColor
    property alias image: image
    width: 800
    height: 600

    Image {
        id: image
        x: 0
        y: 1
        width: 307
        height: 604
        source: "../../../Downloads/dws/Flowers_02_4K.jpg"
        fillMode: Image.PreserveAspectCrop
    }

    TextField {
        id: textField
        x: 318
        y: 48
        width: 348
        height: 43
        text: qsTr("")
        font.pointSize: 12
        placeholderText: qsTr("UserName")
    }
}
