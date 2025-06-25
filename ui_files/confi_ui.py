# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_files/config-2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(858, 439)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("../assets/config-icon.png"),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )
        Dialog.setWindowIcon(icon)
        self.formLayout = QtWidgets.QFormLayout(Dialog)
        self.formLayout.setObjectName("formLayout")
        self.toolBox = QtWidgets.QToolBox(Dialog)
        self.toolBox.setStyleSheet(
            "QToolBox::tab {\n"
            "    background: #E6E6E6;\n"
            "    border-radius: 5px;\n"
            "    color: black;\n"
            "}\n"
            "QToolBox::tab:selected { /* italicize selected tabs */\n"
            "    font: italic;\n"
            "    color: black;\n"
            "    border-radius: 5px;\n"
            "    background: #c6c6c6;\n"
            "}"
        )
        self.toolBox.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 840, 208))
        self.page.setObjectName("page")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_5.setObjectName("gridLayout_5")
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_5.addItem(spacerItem, 10, 0, 1, 1)
        self.exp_name = QtWidgets.QLineEdit(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exp_name.sizePolicy().hasHeightForWidth())
        self.exp_name.setSizePolicy(sizePolicy)
        self.exp_name.setMaximumSize(QtCore.QSize(171, 1000))
        self.exp_name.setText("")
        self.exp_name.setObjectName("exp_name")
        self.gridLayout_5.addWidget(self.exp_name, 10, 1, 1, 1)
        self.cvat_api_button = QtWidgets.QRadioButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cvat_api_button.sizePolicy().hasHeightForWidth()
        )
        self.cvat_api_button.setSizePolicy(sizePolicy)
        self.cvat_api_button.setObjectName("cvat_api_button")
        self.gridLayout_5.addWidget(
            self.cvat_api_button, 10, 3, 1, 1, QtCore.Qt.AlignHCenter
        )
        self.label_4 = QtWidgets.QLabel(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_5.addWidget(self.label_4, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_5.addWidget(self.label_8, 1, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_5.addItem(spacerItem1, 19, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_5.addItem(spacerItem2, 0, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_5.addItem(spacerItem3, 10, 2, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_5.addItem(spacerItem4, 10, 5, 1, 1)
        self.general_button = QtWidgets.QRadioButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.general_button.sizePolicy().hasHeightForWidth()
        )
        self.general_button.setSizePolicy(sizePolicy)
        self.general_button.setObjectName("general_button")
        self.gridLayout_5.addWidget(self.general_button, 18, 3, 1, 1)
        self.cvat_manual_button = QtWidgets.QRadioButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.cvat_manual_button.sizePolicy().hasHeightForWidth()
        )
        self.cvat_manual_button.setSizePolicy(sizePolicy)
        self.cvat_manual_button.setObjectName("cvat_manual_button")
        self.gridLayout_5.addWidget(self.cvat_manual_button, 11, 3, 1, 1)
        self.toolBox.addItem(self.page, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 572, 74))
        self.page_2.setObjectName("page_2")
        self.gridLayout = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem5 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout.addItem(spacerItem5, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 1, QtCore.Qt.AlignHCenter)
        spacerItem6 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout.addItem(spacerItem6, 2, 4, 1, 1)
        self.browse = QtWidgets.QPushButton(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browse.sizePolicy().hasHeightForWidth())
        self.browse.setSizePolicy(sizePolicy)
        self.browse.setAutoDefault(False)
        self.browse.setObjectName("browse")
        self.gridLayout.addWidget(self.browse, 2, 3, 1, 1)
        self.data_dir = QtWidgets.QLineEdit(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.data_dir.sizePolicy().hasHeightForWidth())
        self.data_dir.setSizePolicy(sizePolicy)
        self.data_dir.setMinimumSize(QtCore.QSize(300, 0))
        self.data_dir.setText("")
        self.data_dir.setObjectName("data_dir")
        self.gridLayout.addWidget(self.data_dir, 2, 2, 1, 1)
        self.data_format = QtWidgets.QComboBox(self.page_2)
        self.data_format.setMinimumSize(QtCore.QSize(150, 0))
        self.data_format.setObjectName("data_format")
        self.gridLayout.addWidget(self.data_format, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        spacerItem7 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout.addItem(spacerItem7, 0, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout.addItem(spacerItem8, 3, 1, 1, 1)
        self.toolBox.addItem(self.page_2, "")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setGeometry(QtCore.QRect(0, 0, 383, 74))
        self.page_3.setObjectName("page_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_6 = QtWidgets.QLabel(self.page_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.model_name = QtWidgets.QComboBox(self.page_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.model_name.sizePolicy().hasHeightForWidth())
        self.model_name.setSizePolicy(sizePolicy)
        self.model_name.setMinimumSize(QtCore.QSize(200, 0))
        self.model_name.setObjectName("model_name")
        self.gridLayout_2.addWidget(self.model_name, 3, 1, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem9, 3, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_2.addItem(spacerItem10, 5, 1, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_2.addItem(spacerItem11, 0, 1, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem12, 3, 4, 1, 1)
        self.use_cuda = QtWidgets.QCheckBox(self.page_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.use_cuda.sizePolicy().hasHeightForWidth())
        self.use_cuda.setSizePolicy(sizePolicy)
        self.use_cuda.setObjectName("use_cuda")
        self.gridLayout_2.addWidget(self.use_cuda, 3, 3, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem13, 3, 2, 1, 1)
        self.toolBox.addItem(self.page_3, "")
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setGeometry(QtCore.QRect(0, 0, 664, 101))
        self.page_4.setObjectName("page_4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_4)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.labels_to_classes = QtWidgets.QComboBox(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labels_to_classes.sizePolicy().hasHeightForWidth()
        )
        self.labels_to_classes.setSizePolicy(sizePolicy)
        self.labels_to_classes.setMinimumSize(QtCore.QSize(150, 0))
        self.labels_to_classes.setObjectName("labels_to_classes")
        self.gridLayout_3.addWidget(self.labels_to_classes, 2, 2, 1, 1)
        self.add_labels = QtWidgets.QPushButton(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_labels.sizePolicy().hasHeightForWidth())
        self.add_labels.setSizePolicy(sizePolicy)
        self.add_labels.setMinimumSize(QtCore.QSize(0, 0))
        self.add_labels.setObjectName("add_labels")
        self.gridLayout_3.addWidget(self.add_labels, 2, 3, 1, 1)
        self.clear_labels = QtWidgets.QPushButton(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clear_labels.sizePolicy().hasHeightForWidth())
        self.clear_labels.setSizePolicy(sizePolicy)
        self.clear_labels.setObjectName("clear_labels")
        self.gridLayout_3.addWidget(self.clear_labels, 2, 4, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 1, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_5 = QtWidgets.QLabel(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.labels = QtWidgets.QLineEdit(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labels.sizePolicy().hasHeightForWidth())
        self.labels.setSizePolicy(sizePolicy)
        self.labels.setMaximumSize(QtCore.QSize(100, 25))
        self.labels.setObjectName("labels")
        self.gridLayout_3.addWidget(self.labels, 2, 1, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_3.addItem(spacerItem14, 3, 2, 1, 1)
        self.scrollArea = QtWidgets.QScrollArea(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(200, 0))
        self.scrollArea.setMaximumSize(QtCore.QSize(16777215, 50))
        self.scrollArea.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 98, 32))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.scrollAreaWidgetContents.sizePolicy().hasHeightForWidth()
        )
        self.scrollAreaWidgetContents.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labels_to_classes_list = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labels_to_classes_list.setText("")
        self.labels_to_classes_list.setObjectName("labels_to_classes_list")
        self.verticalLayout.addWidget(self.labels_to_classes_list)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_3.addWidget(self.scrollArea, 2, 5, 1, 1)
        spacerItem15 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_3.addItem(spacerItem15, 0, 1, 1, 1)
        spacerItem16 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_3.addItem(spacerItem16, 2, 0, 1, 1)
        spacerItem17 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_3.addItem(spacerItem17, 2, 6, 1, 1)
        self.toolBox.addItem(self.page_4, "")
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setGeometry(QtCore.QRect(0, 0, 336, 124))
        self.page_5.setObjectName("page_5")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.page_5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_15 = QtWidgets.QLabel(self.page_5)
        self.label_15.setObjectName("label_15")
        self.gridLayout_4.addWidget(self.label_15, 3, 1, 1, 1)
        self.query_mode = QtWidgets.QComboBox(self.page_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.query_mode.sizePolicy().hasHeightForWidth())
        self.query_mode.setSizePolicy(sizePolicy)
        self.query_mode.setObjectName("query_mode")
        self.gridLayout_4.addWidget(self.query_mode, 4, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.page_5)
        self.label_19.setObjectName("label_19")
        self.gridLayout_4.addWidget(self.label_19, 3, 0, 1, 1)
        self.performance_thresh = QtWidgets.QSlider(self.page_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.performance_thresh.sizePolicy().hasHeightForWidth()
        )
        self.performance_thresh.setSizePolicy(sizePolicy)
        self.performance_thresh.setMaximum(100)
        self.performance_thresh.setSingleStep(1)
        self.performance_thresh.setOrientation(QtCore.Qt.Horizontal)
        self.performance_thresh.setObjectName("performance_thresh")
        self.gridLayout_4.addWidget(self.performance_thresh, 4, 1, 1, 1)
        spacerItem18 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_4.addItem(spacerItem18, 0, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.page_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setObjectName("label_16")
        self.gridLayout_4.addWidget(self.label_16, 1, 0, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.page_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 1, 1, 1, 1)
        self.epochs = QtWidgets.QSpinBox(self.page_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.epochs.sizePolicy().hasHeightForWidth())
        self.epochs.setSizePolicy(sizePolicy)
        self.epochs.setObjectName("epochs")
        self.gridLayout_4.addWidget(self.epochs, 2, 0, 1, 1)
        self.subset_size = QtWidgets.QSpinBox(self.page_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.subset_size.sizePolicy().hasHeightForWidth())
        self.subset_size.setSizePolicy(sizePolicy)
        self.subset_size.setObjectName("subset_size")
        self.gridLayout_4.addWidget(self.subset_size, 2, 2, 1, 1)
        self.batch_size = QtWidgets.QSpinBox(self.page_5)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.batch_size.sizePolicy().hasHeightForWidth())
        self.batch_size.setSizePolicy(sizePolicy)
        self.batch_size.setObjectName("batch_size")
        self.gridLayout_4.addWidget(self.batch_size, 2, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.page_5)
        self.label_17.setObjectName("label_17")
        self.gridLayout_4.addWidget(self.label_17, 1, 2, 1, 1)
        spacerItem19 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_4.addItem(spacerItem19, 5, 1, 1, 1)
        self.toolBox.addItem(self.page_5, "")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.toolBox)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.import_config = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.import_config.sizePolicy().hasHeightForWidth()
        )
        self.import_config.setSizePolicy(sizePolicy)
        self.import_config.setMinimumSize(QtCore.QSize(300, 0))
        self.import_config.setObjectName("import_config")
        self.horizontalLayout.addWidget(self.import_config)
        self.first_ok = QtWidgets.QPushButton(Dialog)
        self.first_ok.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.first_ok.sizePolicy().hasHeightForWidth())
        self.first_ok.setSizePolicy(sizePolicy)
        self.first_ok.setMinimumSize(QtCore.QSize(300, 0))
        self.first_ok.setObjectName("first_ok")
        self.horizontalLayout.addWidget(self.first_ok)
        self.formLayout.setLayout(
            3, QtWidgets.QFormLayout.SpanningRole, self.horizontalLayout
        )

        self.retranslateUi(Dialog)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Configuration"))
        self.cvat_api_button.setText(
            _translate("Dialog", "CVAT (API-based Ingegration)")
        )
        self.label_4.setText(_translate("Dialog", "Name"))
        self.label_8.setText(_translate("Dialog", "Manual Annotation Tool"))
        self.general_button.setText(
            _translate("Dialog", "General (Manual Integration)")
        )
        self.cvat_manual_button.setText(
            _translate("Dialog", "CVAT (Manual Integration)")
        )
        self.toolBox.setItemText(
            self.toolBox.indexOf(self.page), _translate("Dialog", "Experiment")
        )
        self.label_2.setText(_translate("Dialog", "Data Directory"))
        self.browse.setText(_translate("Dialog", "Browse"))
        self.label_3.setText(_translate("Dialog", "Data Format"))
        self.toolBox.setItemText(
            self.toolBox.indexOf(self.page_2), _translate("Dialog", "Dataset")
        )
        self.label_6.setText(_translate("Dialog", "Model"))
        self.use_cuda.setText(_translate("Dialog", "Enable Cuda"))
        self.toolBox.setItemText(
            self.toolBox.indexOf(self.page_3), _translate("Dialog", "Model")
        )
        self.add_labels.setText(_translate("Dialog", "Add"))
        self.clear_labels.setText(_translate("Dialog", "Clear"))
        self.label_7.setText(_translate("Dialog", "Classes"))
        self.label_5.setText(_translate("Dialog", "Labels"))
        self.toolBox.setItemText(
            self.toolBox.indexOf(self.page_4), _translate("Dialog", "Labels")
        )
        self.label_15.setText(_translate("Dialog", "Performance Threshold"))
        self.label_19.setText(_translate("Dialog", "Query Mode"))
        self.label_16.setText(_translate("Dialog", "Training Epochs"))
        self.label_18.setText(_translate("Dialog", "Training Batch Size"))
        self.label_17.setText(_translate("Dialog", "Subset Size"))
        self.toolBox.setItemText(
            self.toolBox.indexOf(self.page_5),
            _translate("Dialog", "Advanced Parameters"),
        )
        self.import_config.setText(_translate("Dialog", "Import Config"))
        self.first_ok.setText(_translate("Dialog", "OK"))
