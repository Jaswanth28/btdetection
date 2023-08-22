import QtQuick
import QtQuick.Window

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    Image {
        id: image
        x: 0
        y: 0
        width: 226
        height: 480
        source: "../../Downloads/dws/Flowers_02_4K.jpg"
        fillMode: Image.PreserveAspectCrop
    }

}
