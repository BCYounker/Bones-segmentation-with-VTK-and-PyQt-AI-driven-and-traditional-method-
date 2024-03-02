import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import pandas as pd
import os
import subprocess

import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from vtkmodules.all import (
    vtkRenderer, vtkImageThreshold, vtkMarchingCubes,
    vtkSmoothPolyDataFilter, vtkPolyDataNormals, vtkPolyDataMapper, vtkActor,
    vtkInteractorStyleTrackballCamera, vtkCellPicker
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from testvtkgui import Ui_MainWindow

#from vtkmodules.util import *
import time

class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('ProjectChallenge-YBC')

        self.init2D = False
        self.dataExtent = []
        self.dataDimensions = []

        #Select Path
        SelectDir = QAction("Select a directory", self)
        SelectDir.triggered.connect(self.on_SelectDicom_clicked)
        self.toolBar.addAction(SelectDir)

        #Main VTK widget
        self.frame = QtWidgets.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.formLayout.addWidget(self.vtkWidget) # create widget for layout

        self.SegImageData = vtk.vtkImageData()
        # self.ThresImageData = vtk.vtkImageData()
        self.OrgImageData = vtk.vtkImageData()
        self.PolyData = vtk.vtkPolyData()
        self.ResultPolyData1 = vtk.vtkPolyData()
        self.ResultPolyData2 = vtk.vtkPolyData()

        ##########vtkMain
        self.ren = vtk.vtkRenderer()
        self.RenWin = self.vtkWidget.GetRenderWindow()
        self.RenWin.AddRenderer(self.ren)
        self.RenWin.Render()

        ##########vtkResult1
        self.renResult1 = vtk.vtkRenderer()
        self.RenWinResult1 = self.widget_2.GetRenderWindow()
        self.RenWinResult1.AddRenderer(self.renResult1)
        self.RenWinResult1.Render()

        # ##########vtkResult2
        self.renResult2 = vtk.vtkRenderer()
        self.RenWinResult2 = self.widget_3.GetRenderWindow()
        self.RenWinResult2.AddRenderer(self.renResult2)
        self.RenWinResult2.Render()

        ##########vtkTop
        self.renTop = vtk.vtkRenderer()
        self.RenWinTop = self.widget_vtkTop.GetRenderWindow()
        self.RenWinTop.AddRenderer(self.renTop)
        self.RenWinTop.Render()

        ##########vtkMid
        self.renMid = vtk.vtkRenderer()
        self.RenWinMid = self.widget_vtkMid.GetRenderWindow()
        self.RenWinMid.AddRenderer(self.renMid)
        self.RenWinMid.Render()

        ##########vtkBot
        self.renBottom = vtk.vtkRenderer()
        self.RenWinBottom  = self.widget_vtkBottom.GetRenderWindow()
        self.RenWinBottom .AddRenderer(self.renBottom )
        self.RenWinBottom .Render()

        ##############  3D Interaction

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(interactorStyle)

        self.irenResult1 = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.irenResult1.SetInteractorStyle(interactorStyle)

        # Setting clicked event from mouse
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.iren.SetPicker(self.picker)
        self.iren.AddObserver("LeftButtonPressEvent", self.on_left_button_press, 1.0)
        self.iren.Initialize()
        self.iren.Start()
        self.selectedPoints = []
        self.sphereActors = []

        self.checkBox.stateChanged.connect(self.toggleVolumeRender)
        self.populateComboBoxes()
        self.comboBox_1.currentIndexChanged.connect(self.on_comboBox_1_indexChanged)
        self.comboBox_2.currentIndexChanged.connect(self.on_comboBox_2_indexChanged)
        self.volumnRender=True

        self.UpThresholdValue = 1000
        self.LowerThresholdValue =167#270#335#200
        self.MarchingcubeValue = 250
        # self.segparam=1

        self.ActorsColorCollect = []
        self.actor_collection = []

        self.horizontalSliderLower.setMaximum(4000)
        self.horizontalSliderLower.setMinimum(0)
        self.horizontalSliderUp.setMaximum(4000)
        self.horizontalSlider.setMaximum(500)
        self.horizontalSlider.setMinimum(0)
        # self.horizontalSliderSeg.setMaximum(500)
        # self.horizontalSliderSeg.setMinimum(0)

        self.mapperA = vtk.vtkPolyDataMapper()
        self.actorA = vtk.vtkActor()
        self.mapperB = vtk.vtkPolyDataMapper()
        self.actorB = vtk.vtkActor()
        # vtk.vtkObject.GlobalWarningDisplayOn()

        self.show()

    # Maintain camera settings in interation
    def toggleVolumeRender(self):
        # Save the current camera settings
        camera = self.ren.GetActiveCamera()
        position = camera.GetPosition()
        focalPoint = camera.GetFocalPoint()
        viewUp = camera.GetViewUp()
        zoom = camera.GetParallelScale()

        self.volumeRender = not self.checkBox.isChecked()
        if not self.volumeRender:  # delete volume, add polydata actor
            self.ren.RemoveVolume(self.volume)
            self.ren.AddActor(self.actorA)
        else:
            self.ren.RemoveActor(self.actorA)
            self.ren.AddVolume(self.volume)

        # After adding the new actor or volume, restore the camera settings
        # self.ren.ResetCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(focalPoint)
        camera.SetViewUp(viewUp)
        camera.SetParallelScale(zoom)

        self.RenWin.Render()

    # Generate 3D red ball to select regions
    def on_left_button_press(self, obj, event):
        clickPos = self.iren.GetEventPosition()
        self.picker.Pick(clickPos[0], clickPos[1], 0, self.ren)
        pickedPosition = self.picker.GetPickPosition()
        print(f"Picked 3D position: {pickedPosition}")
        # Store the picked position
        self.selectedPoints.append(pickedPosition)
        if len(self.selectedPoints) > 2:
            self.selectedPoints.pop(0)  # Keep only the latest two points
        self.drawSpheres()
    def drawSpheres(self):
        # Remove previous circles from the renderer
        for actor in self.sphereActors:
            self.ren.RemoveActor(actor)
        self.sphereActors.clear()

        # Draw new spheres for each selected point
        for point in self.selectedPoints:
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(point[0], point[1], point[2])
            sphereSource.SetRadius(1)  # Set the radius of the sphere

            # Create a mapper and actor for the sphere
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Set sphere color

            self.ren.AddActor(actor)
            self.sphereActors.append(actor)

        self.iren.Render()

    @QtCore.pyqtSlot(int)
    def on_verticalSliderTop_valueChanged(self, value):
        self.viewerXY.SetSlice(value)

    @QtCore.pyqtSlot(int)
    def on_verticalSliderMid_valueChanged(self, value):
        self.viewerYZ.SetSlice(value)

    @QtCore.pyqtSlot(int)
    def on_verticalSliderBottom_valueChanged(self, value):
        self.viewerXZ.SetSlice(value)

    # Setup slots for windowing sliders
    @QtCore.pyqtSlot(int)
    def on_WindowCenterSlider_valueChanged(self, value):
        for x in [self.viewerXY, self.viewerXZ, self.viewerYZ]:
            x.SetColorLevel(value)
            x.Render()

    @QtCore.pyqtSlot(int)
    def on_WindowWidthSlider_valueChanged(self, value):
        for x in [self.viewerXY, self.viewerXZ, self.viewerYZ]:
            x.SetColorWindow(value)
            x.Render()

    #@QtCore.pyqtSlot()
    def on_SelectDicom_clicked(self):
        dicompath = QtWidgets.QFileDialog.getExistingDirectory(None, "Select data directory", "")
        self.dicompath=dicompath
        if dicompath == "":
            print("\n No valid directory")
            return
        try:
            self.OrgImageData.DeepCopy(self.GetDicmoDataFromPath(dicompath))
            self.loadImageData(dicompath)
        except:
            infobox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Error", "Load data failed")
            infobox.exec_()

    # @QtCore.pyqtSlot(int)
    # def on_horizontalSliderUp_valueChanged(self, value):
    #     self.label_up.setText(str(value))
    #     self.UpThresholdValue = value
    #     self.changeThresholdValue()
    #     #self.on_pushButtonSegment_clicked()
    @QtCore.pyqtSlot(int)
    def on_horizontalSliderUp_valueChanged(self, value):
        roundedValue = 50 * round(value / 50)

        # Set the slider value to the rounded value
        if value != roundedValue:
            self.horizontalSliderUp.setValue(roundedValue)
            return
        self.label_up.setText(str(roundedValue))
        self.UpThresholdValue = roundedValue
        # self.changeThresholdValue()

    @QtCore.pyqtSlot(int)
    def on_horizontalSliderLower_valueChanged(self, value):
        self.label_lower.setText(str(value))
        self.LowerThresholdValue = value
        # self.changeThresholdValue()
        # self.on_pushButtonSegment_clicked()

    @QtCore.pyqtSlot(int)
    def on_horizontalSlider_valueChanged(self, value):
        self.label_cubes.setText(str(value))
        self.MarchingcubeValue = value


    @QtCore.pyqtSlot()
    def on_ThresholdSeg_clicked(self):
        self.changeThresholdValue()
        # create the mapper
        self.mapperA.SetInputData(self.PolyData)
        self.renResult1.RemoveVolume(self.volume)
        # self.ren.RemoveActor(self.actorA)
        self.actorA.SetMapper(self.mapperA)
        self.ren.ResetCamera()
        self.RenWin.Render()

    # Call region growing
    @QtCore.pyqtSlot()
    def on_RegionGrowing_clicked(self):
        self.changeSegValueContour()
        self.ren.ResetCamera()
        self.RenWin.Render()

    # Select and Render 2 regions
    @QtCore.pyqtSlot()
    def on_Select2_clicked(self):
        polyConnectivity = vtk.vtkPolyDataConnectivityFilter()
        polyConnectivity.SetInputData(self.PolyData)
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(self.PolyData)  # Assuming self.PolyData is the output from your connectivity filter
        pointLocator.BuildLocator()
        regionIDs = []
        scalars = self.PolyData.GetPointData().GetScalars()  # This should contain the region IDs as scalar values
        index = 0
        for point in self.selectedPoints:
            pointId = pointLocator.FindClosestPoint(point)
            regionID = scalars.GetValue(pointId)
            print("Region ID for selected point:", regionID)
            regionIDs.append(regionID)
            polyConnectivity.InitializeSpecifiedRegionList()  # Reset the list of specified regions
            polyConnectivity.AddSpecifiedRegion(regionID)
            polyConnectivity.SetExtractionModeToSpecifiedRegions()
            polyConnectivity.Update()
            if index == 0:
                self.ResultPolyData1.DeepCopy(polyConnectivity.GetOutput())
                index += 1
            else:
                self.ResultPolyData2.DeepCopy(polyConnectivity.GetOutput())
        if self.ResultPolyData1:
            self.renderPolyData(self.ResultPolyData1, self.renResult1, self.RenWinResult1)
        if self.ResultPolyData2:
            self.renderPolyData(self.ResultPolyData2, self.renResult2, self.RenWinResult2)

    @QtCore.pyqtSlot()
    def on_ExportSTL_clicked(self):
        if self.ResultPolyData1:
            stlWriter = vtk.vtkSTLWriter()
            stlWriter.SetInputData(self.ResultPolyData1)
            stlWriter.SetFileName(f"./segmentationManual/seg_1.stl")
            stlWriter.Write()

        if self.ResultPolyData2:
            stlWriter = vtk.vtkSTLWriter()
            stlWriter.SetInputData(self.ResultPolyData2)
            stlWriter.SetFileName(f"./segmentationManual/seg_2.stl")
            stlWriter.Write()

    # AI inference
    @QtCore.pyqtSlot()
    def on_ExportSTLAI_clicked(self):
        try:
            import torch
            device = "gpu" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        QtWidgets.QMessageBox.information(self, "Processing","Setting up environment and processing data. This may take a few minutes for first time. Press OK to continue.")

        # Run TotalSegmentator command
        input_path = self.dicompath
        output_path = "./segmentationsAI"
        roi_subset = f"{self.structure_name_1} {self.structure_name_2}" #try calivicula_right
        command = f"TotalSegmentator -i {input_path} -o {output_path} --device {device} --roi_subset {roi_subset}"
        try:
            result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)#,encoding='utf-8')
            output = result.stdout
            if output:
                QtWidgets.QMessageBox.information(self, "Success", f"Command executed successfully:\n{output}\n Press OK to export STL and Render")
        except subprocess.CalledProcessError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to execute TotalSegmentator: {e}")

        self.processOutputFiles(output_path)

    def processOutputFiles(self, output_path):
        index=0
        # Loop through all NIfTI files in the output directory
        for file_name in os.listdir(output_path):
            if file_name.endswith(".nii.gz"):
                # Extract the base name of the file without the extension
                base_file_name = os.path.splitext(os.path.splitext(file_name)[0])[0]
                # print(base_file_name)

                # Check if the base file name matches either self.structure_name_1 or self.structure_name_2
                if base_file_name == self.structure_name_1 or base_file_name == self.structure_name_2:
                    nii_file_path = os.path.join(output_path, file_name)
                    # Determine the STL file name based on the NIfTI file name
                    stl_file_name = base_file_name + ".stl"
                    polydata=self.NiiToSTL(output_path, nii_file_path, stl_file_name) #try add volumn rendering
                    if index==0:
                        self.ResultPolyData1 = polydata
                        index += 1
                        # self.ResultVolumnData1 = volumedata
                    else:
                        self.ResultPolyData2 = polydata


        if self.ResultPolyData1:
            self.renderPolyData(self.ResultPolyData1, self.renResult1, self.RenWinResult1)
        if self.ResultPolyData2:
            self.renderPolyData(self.ResultPolyData2, self.renResult2, self.RenWinResult2)

    def NiiToSTL(self, output_path, nii_file_path, stl_file_name):
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(nii_file_path)
        reader.Update()

        # imageResample = vtk.vtkImageResample()
        # imageResample.SetInputConnection(reader.GetOutputPort())
        # imageResample.SetInterpolationModeToCubic()
        # imageResample.SetAxisMagnificationFactor(0, 4)
        # imageResample.SetAxisMagnificationFactor(1, 4)
        # imageResample.SetAxisMagnificationFactor(2, 4)
        # imageResample.Update()
        #
        # #optional
        # gaussianSmoothFilter = vtk.vtkImageGaussianSmooth()
        # gaussianSmoothFilter.SetInputConnection(imageResample.GetOutputPort())
        # gaussianSmoothFilter.SetRadiusFactors(0.1, 0.1, 0.1)  # Adjust the radius factors as needed
        # gaussianSmoothFilter.Update()

        marchingCubes = vtk.vtkMarchingCubes()
        marchingCubes.SetInputConnection(reader.GetOutputPort())
        marchingCubes.SetValue(0, 0.5)  # Threshold adjustment may be needed

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marchingCubes.GetOutputPort())
        smoother.SetNumberOfIterations(50)  # Adjust as needed

        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetInputConnection(smoother.GetOutputPort())
        stl_file_path = os.path.join(output_path, stl_file_name)
        stlWriter.SetFileName(stl_file_path)
        stlWriter.Write()
        return smoother.GetOutput()

    def renderPolyData(self, polyData, renderer, renderWindow):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.RemoveAllViewProps()
        renderer.AddActor(actor)
        renderer.ResetCamera()
        renderWindow.Render()

    @QtCore.pyqtSlot(int)
    def on_comboBox_1_indexChanged(self, index):
        self.structure_name_1 = self.comboBox_1.itemText(index)
        print(f"ComboBox1 selected structure: {self.structure_name_1}")

    @QtCore.pyqtSlot(int)
    def on_comboBox_2_indexChanged(self, index):
        self.structure_name_2 = self.comboBox_2.itemText(index)
        print(f"ComboBox2 selected structure: {self.structure_name_2}")

    # Display classes name for AI segmentation
    def populateComboBoxes(self):
        file_path = './totalsegmentator_mapping.csv'
        data = pd.read_csv(file_path)
        structures = data['Structure'].tolist()

        self.comboBox_1.addItems(structures)
        self.comboBox_2.addItems(structures)

        default_value1 = "humerus_right"
        default_value2 = "scapula_right"
        self.structure_name_1="humerus_right"
        self.structure_name_2 = "scapula_right"
        default_index1 = structures.index(default_value1) if default_value1 in structures else 0
        default_index2 = structures.index(default_value2) if default_value2 in structures else 0
        self.comboBox_1.setCurrentIndex(default_index1)
        self.comboBox_2.setCurrentIndex(default_index2)

    # Volumn Render Settings
    def setupVolumeRenderingFromImageData(self,imagedata):
        # Opacity transfer function
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0.00)
        opacityTransferFunction.AddPoint(300, 0.7)
        opacityTransferFunction.AddPoint(478, 1)
        opacityTransferFunction.AddPoint(1500, 1)

        # Color transfer function
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(100, 220 / 255, 21 / 255, 3 / 255)
        colorTransferFunction.AddRGBPoint(300, 250 / 255, 1, 189 / 255)
        colorTransferFunction.AddRGBPoint(478, 224 / 255, 1, 249 / 255)
        colorTransferFunction.AddRGBPoint(1500, 1, 1, 1)

        # Volume property
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()

        # Volume mapper
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(imagedata)

        # Volume actor
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        return volume

    # Surface Rendering
    def setupContourFilterFromImageData(self, imagedata):
        #actor for polydata
        contoursA = vtk.vtkContourFilter()
        contoursA.SetInputData(imagedata)
        contoursA.SetValue(0, 1)
        # contoursA.GenerateValues(5,0.,5.)
        contoursA.Update()
        # self.PolyData.DeepCopy(contoursA.GetOutput())
        self.mapperA.ScalarVisibilityOff()
        self.mapperA.SetInputData(contoursA.GetOutput())
        # self.mapperA.SetScalarRange(1., 5.)  range of scalar values to map to colors.
        propA = vtk.vtkProperty()
        propA.SetColor(1.0, 1.0, 1.0)
        propA.SetOpacity(1)
        self.actorA.SetProperty(propA)
        self.actorA.SetMapper(self.mapperA)


    def loadImageData(self,path):
        # create the filter GetDicmoDataFromPath
        if False==self.init2D:
            self.Init2DView()

        # surface rendering
        self.setupContourFilterFromImageData(self.OrgImageData)

        # Volumn rendering
        self.volume = self.setupVolumeRenderingFromImageData(self.OrgImageData)
        self.volume = self.setupVolumeRenderingFromImageData(self.OrgImageData)
        self.ren.AddVolume(self.volume)
        self.ren.ResetCamera()
        self.RenWin.Render()

        self.dataExtent = self.OrgImageData.GetExtent()
        print(f'Shape of input data, X:{self.dataExtent[1]},Y:{self.dataExtent[3]},Z:{self.dataExtent[5]}')
        # Get data range
        self.dataRange = self.OrgImageData.GetScalarRange()
        # self.dataRange.
        print(f'Range of CT numbers{self.dataRange[0]} to {self.dataRange[1]} ')
        self.show_Data()


    def GetDicmoDataFromPath(self,path):
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(path)
        reader.Update()

        return reader.GetOutput()

    #区域增长分割（连接性）基于面 version 9
    def changeSegValueContour(self):
        # cleanFilter = vtk.vtkCleanPolyData()
        # cleanFilter.SetInputData(self.PolyData)
        # cleanFilter.PointMergingOn()
        # # cleanFilter.SetTolerance(0.01)
        # cleanFilter.Update()

        # Step 2: Extract all regions
        polyConnectivity = vtk.vtkPolyDataConnectivityFilter()
        #polyConnectivity.SetInputData(cleanFilter.GetOutput())
        polyConnectivity.SetInputData(self.PolyData)
        polyConnectivity.SetExtractionModeToAllRegions()
        polyConnectivity.ColorRegionsOn()
        polyConnectivity.Update()

        numberOfRegions = polyConnectivity.GetNumberOfExtractedRegions()
        print('Number of regions:',numberOfRegions)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(max(numberOfRegions, 10))
        lut.Build()
        self.randomColors(lut, numberOfRegions)
        self.mapperA.ScalarVisibilityOn()
        #self.mapperA.SetInputConnection(polyConnectivity.GetOutputPort())
        self.mapperA.SetInputData(polyConnectivity.GetOutput())
        self.mapperA.SetScalarRange(polyConnectivity.GetOutput().GetPointData().GetScalars().GetRange())
        self.mapperA.SetLookupTable(lut)
        self.actorA.SetMapper(self.mapperA)
        self.PolyData.DeepCopy(polyConnectivity.GetOutput())


        # regionSizes = polyConnectivity.GetRegionSizes()
        # # Step 3: Filter regions by size
        # sizeThreshold = 800  # Define your size threshold here
        # regionsToKeep = []
        # for i in range(regionSizes.GetNumberOfTuples()):
        #     regionSize = regionSizes.GetValue(i)
        #     if regionSize > sizeThreshold:
        #         regionsToKeep.append(i)
        # print(f'Keeped regions {regionsToKeep}')

        # # Step 4: Merge
        # mergedPolyData = vtk.vtkPolyData()
        # appendFilter = vtk.vtkAppendPolyData()
        # polyConnectivity_1 = vtk.vtkPolyDataConnectivityFilter()
        # polyConnectivity_1.SetInputData(self.PolyData)
        # for regionId in regionsToKeep:
        #     polyConnectivity_1.AddSpecifiedRegion(int(regionId))  # Ensure regionId is an int
        # polyConnectivity_1.SetExtractionModeToSpecifiedRegions()
        # polyConnectivity_1.Update()
        # appendFilter.AddInputData(polyConnectivity_1.GetOutput())
        # appendFilter.Update()
        # mergedPolyData.DeepCopy(appendFilter.GetOutput())

        # # # Extracted data for debugging
        # # extractedData = polyConnectivity_1.GetOutput()
        # # print(f"Extracted Data Points: {extractedData.GetNumberOfPoints()}")
        # # print(f"Extracted Data Cells: {extractedData.GetNumberOfCells()}")
        # # stlWriter = vtk.vtkSTLWriter()
        # # stlWriter.SetInputData(mergedPolyData)
        # # stlWriter.SetFileName(f"./region_x.stl")
        # # stlWriter.Write()

    # Threshold based segmentation with many optimisations
    def changeThresholdValue(self):
        self.ImageThreshold = vtk.vtkImageThreshold()
        self.ImageThreshold.SetInputData(self.OrgImageData)
        self.ImageThreshold.ThresholdBetween(self.LowerThresholdValue, self.UpThresholdValue)
        # self.ImageThreshold.SetOutValue(-1024)
        # self.ImageThreshold.SetInValue(1024)
        self.ImageThreshold.Update()

        erosionFilter = vtk.vtkImageDilateErode3D()
        erosionFilter.SetInputData(self.ImageThreshold.GetOutput())
        erosionFilter.SetErodeValue(1024)  # Foreground value to erode
        erosionFilter.SetDilateValue(-1024)  # Background value
        erosionFilter.SetKernelSize(2, 2, 2)  # Adjust as needed for erosion
        erosionFilter.Update()

        dilationFilter = vtk.vtkImageDilateErode3D()
        dilationFilter.SetInputData(erosionFilter.GetOutput())
        dilationFilter.SetDilateValue(1024)
        dilationFilter.SetErodeValue(-1024)
        dilationFilter.SetKernelSize(2, 2, 2)  # Adjust the kernel size as needed, possibly different from erosion
        dilationFilter.Update()

        # 2. Gaussian Smoothing
        gaussianSmoothFilter = vtk.vtkImageGaussianSmooth()
        gaussianSmoothFilter.SetInputConnection(dilationFilter.GetOutputPort())
        gaussianSmoothFilter.SetRadiusFactors(0.1, 0.1, 0.1)  # Adjust the radius factors as needed
        gaussianSmoothFilter.Update()

        # 3. Extracting Surface
        MarchingCub = vtk.vtkMarchingCubes()
        MarchingCub.SetInputData(gaussianSmoothFilter.GetOutput())
        # MarchingCub.SetInputData(dilationFilter.GetOutput())
        # MarchingCub.SetInputData(gaussianSmoothFilter.GetOutput())
        MarchingCub.SetValue(0, self.MarchingcubeValue)
        MarchingCub.Update()

        # 4. subdivision
        # subdivisionFilter = vtk.vtkLoopSubdivisionFilter()
        # subdivisionFilter.SetInputData(MarchingCub.GetOutput())
        # subdivisionFilter.SetNumberOfSubdivisions(1)  # Increase for more detail
        # subdivisionFilter.Update()

        # 5. Surface Smoothing
        smooth = vtk.vtkSmoothPolyDataFilter()
        # smooth.SetInputConnection(subdivisionFilter.GetOutputPort())
        smooth.SetInputConnection(MarchingCub.GetOutputPort())
        smooth.SetNumberOfIterations(30)  # Adjust iteration
        smooth.Update()

        # 6. Hole Filling
        fillHolesFilter = vtk.vtkFillHolesFilter()
        fillHolesFilter.SetInputConnection(smooth.GetOutputPort())
        fillHolesFilter.SetHoleSize(
            fillHolesFilter.GetHoleSizeMaxValue())
        # fillHolesFilter.SetHoleSize(1000)
        fillHolesFilter.Update()

        # 7. Normals Calculation
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(fillHolesFilter.GetOutputPort())
        # normals.ComputePointNormalsOn()
        # normals.ComputeCellNormalsOn()
        normals.SetFeatureAngle(50) # 调整特征角度
        normals.Update()

        self.PolyData.DeepCopy(normals.GetOutput())


    def hex_to_rgb(self,hex_color):
        """
        Convert a hex color string to an RGB tuple.
        """
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        return tuple(int(hex_color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))

    def randomColors(self, lut, numberOfColors):
        # Custom colors defined by hex codes
        custom_hex_colors = [
            "#9e0142", "#d53e4f", "#f46d43", "#fdae61",
            "#fee08b", "#e6f598", "#abdda4", "#abdda4",
            "#3288bd", "#5e4fa2"
        ]

        # Convert hex colors to RGB
        custom_rgb_colors = [self.hex_to_rgb(color) for color in custom_hex_colors]

        # Add custom colors for the first few regions
        for i, rgb_color in enumerate(custom_rgb_colors):
            # Set color with an alpha value of 1.0 (fully opaque)
            lut.SetTableValue(i, *rgb_color, 0.95)

        # Generate random colors for additional regions if necessary
        if numberOfColors > len(custom_rgb_colors):
            randomSequence = vtk.vtkMinimalStandardRandomSequence()
            randomSequence.SetSeed(4355412)  # Use a fixed seed for reproducibility
            for i in range(len(custom_rgb_colors), numberOfColors):
                r = randomSequence.GetRangeValue(0.6, 1.0)
                randomSequence.Next()
                g = randomSequence.GetRangeValue(0.6, 1.0)
                randomSequence.Next()
                b = randomSequence.GetRangeValue(0.6, 1.0)
                randomSequence.Next()
                # Set color with an alpha value of 1.0 (fully opaque)
                lut.SetTableValue(i, r, g, b, 0.95)

    # #区域增长分割（连接性）基于值 version2
    # def changeSegValueContour2(self):
    #     self.ImageThreshold.SetInputData(self.OrgImageData)
    #     self.ImageThreshold.ThresholdBetween(self.LowerThresholdValue, self.UpThresholdValue)
    #     # self.ImageThreshold.SetOutValue(0)
    #     # self.ImageThreshold.SetInValue(255)
    #     self.ImageThreshold.Update()
    #     #self.ThresImageData.DeepCopy(self.ImageThreshold.GetOutput())
    #
    #     # Cast the thresholded image data to UnsignedChar
    #     imageCast = vtk.vtkImageCast()
    #     imageCast.SetInputData(self.ImageThreshold.GetOutput())
    #     imageCast.SetOutputScalarTypeToUnsignedChar()
    #     imageCast.Update()
    #
    #     # use vtkImageSeedConnectivity
    #     seedConnectivity = vtk.vtkImageSeedConnectivity()
    #     seedConnectivity.SetInputData(imageCast.GetOutput())
    #     seedConnectivity.SetInputConnectValue(255)
    #     # set seed points
    #     for point in self.selectedPoints:
    #         # Convert point to index if necessary
    #         seedConnectivity.AddSeed(int(point[0]), int(point[1]), int(point[2]))
    #     seedConnectivity.Update()
    #
    #     # 从阈值和区域生长后的图像生成PolyData
    #     MarchingCubes = vtk.vtkMarchingCubes()
    #     #MarchingCubes.SetInputData(erosionFilter.GetOutput())
    #     # MarchingCubes.SetInputData(dilationFilter.GetOutput())
    #     MarchingCubes.SetInputData(seedConnectivity.GetOutput())
    #     MarchingCubes.SetValue(0, 1)  # 设置等值面提取的阈值
    #     MarchingCubes.Update()
    #
    #     # Smoothing
    #     smoothFilter = vtk.vtkSmoothPolyDataFilter()
    #     smoothFilter.SetInputConnection(MarchingCubes.GetOutputPort())
    #     smoothFilter.SetNumberOfIterations(50)
    #     smoothFilter.Update()
    #
    #     # Calculate Normals
    #     normals = vtk.vtkPolyDataNormals()
    #     normals.SetInputConnection(smoothFilter.GetOutputPort())
    #     normals.SetFeatureAngle(60.0)
    #     normals.Update()
    #
    #     self.PolyData.DeepCopy(normals.GetOutput())
    #
    #     # mapper actor
    #     # self.mapperA.SetInputData(normals.GetOutput())
    #     # self.RenWin.Render()

    # Sliders Viewer
    def Init2DView(self):
        # define 2D viewers
        [self.viewerXY, self.viewerYZ, self.viewerXZ] = [vtk.vtkImageViewer2() for x in range(3)]

        # attach interactors to viewers
        self.viewerXY.SetupInteractor(self.widget_vtkTop)
        self.viewerYZ.SetupInteractor(self.widget_vtkMid)
        self.viewerXZ.SetupInteractor(self.widget_vtkBottom)

        # set render windows for viewers
        self.viewerXY.SetRenderWindow(self.widget_vtkTop.GetRenderWindow())
        self.viewerYZ.SetRenderWindow(self.widget_vtkMid.GetRenderWindow())
        self.viewerXZ.SetRenderWindow(self.widget_vtkBottom.GetRenderWindow())

        # set slicing orientation for viewers
        self.viewerXY.SetSliceOrientationToXZ()
        self.viewerYZ.SetSliceOrientationToYZ()
        self.viewerXZ.SetSliceOrientationToXY()

        self.init2D =True

        # Set up the interaction
        self.interactorXY = self.viewerXY.GetRenderWindow().GetInteractor()
        self.interactorXY.SetInteractorStyle(vtk.vtkInteractorStyleImage())
        self.interactorXZ = self.viewerXZ.GetRenderWindow().GetInteractor()
        self.interactorXZ.SetInteractorStyle(vtk.vtkInteractorStyleImage())
        self.interactorYZ = self.viewerYZ.GetRenderWindow().GetInteractor()
        self.interactorYZ.SetInteractorStyle(vtk.vtkInteractorStyleImage())

    def show_Data(self):
        self.dataExtent = self.OrgImageData.GetExtent()
        print('Re-show the data')
        dataDimensionX = self.dataExtent[1] - self.dataExtent[0]
        dataDimensionY = self.dataExtent[3] - self.dataExtent[2]
        dataDimensionZ = self.dataExtent[5] - self.dataExtent[4]
        self.dataDimensions = [dataDimensionX, dataDimensionY, dataDimensionZ]

        # Calculate index of middle slice
        midslice1 = int((self.dataExtent[1] - self.dataExtent[0]) / 2 + self.dataExtent[0])
        midslice2 = int((self.dataExtent[3] - self.dataExtent[2]) / 2 + self.dataExtent[2])
        midslice3 = int((self.dataExtent[5] - self.dataExtent[4]) / 2 + self.dataExtent[4])

        # Get data range
        self.dataRange = self.OrgImageData.GetScalarRange()

        # Set current slice to the middle one
        for pair in zip([self.viewerXY, self.viewerYZ, self.viewerXZ], [midslice1, midslice2, midslice3]):
            pair[0].SetInputData(self.OrgImageData)
            pair[0].SetSlice(pair[1])
            pair[0].GetRenderer().ResetCamera()
            pair[0].Render()
        pass

        # Set range and proper value for slice sliders
        for pair in zip([self.verticalSliderTop, self.verticalSliderMid, self.verticalSliderBottom, ], self.dataDimensions, [midslice1, midslice2, midslice3]):
            pair[0].setRange(0, pair[1])
            pair[0].setValue(pair[2])

        # # Set windowcenter(brightness) and windowwidth(contrast) for windowing sliders
        self.WindowCenterSlider.setRange(int(self.dataRange[0]), int(self.dataRange[1]))
        self.WindowWidthSlider.setRange(1, int(self.dataRange[1])-int(self.dataRange[0]))

        self.WindowCenterSlider.setValue(int((self.dataRange[0] + self.dataRange[1]) // 2))
        self.WindowWidthSlider.setValue((int(self.dataRange[1])-int(self.dataRange[0])// 2))
        # self.horizontalSliderLower.setValue(self.LowerThresholdValue)   # Set default value for the lower slider
        self.horizontalSliderUp.setValue(self.UpThresholdValue)
        self.horizontalSliderLower.setValue(self.LowerThresholdValue)
        self.horizontalSlider.setValue(self.MarchingcubeValue)

    def closeEvent(self, event):
        self.widget_2.Finalize()
        self.widget_3.Finalize()
        self.vtkWidget.Finalize()
        self.widget_vtkTop.Finalize()
        self.widget_vtkMid.Finalize()
        self.widget_vtkBottom.Finalize()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = Mywindow()
    window.show()
    sys.exit(app.exec_())
