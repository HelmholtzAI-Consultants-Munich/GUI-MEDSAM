import napari
import numpy as np
from napari.layers import Image, Points
from sam2_infer import infer_3d, Point, ClicksInfo
import SimpleITK as sitk


class SegmentationApp:
    def __init__(self):
        self.viewer = napari.Viewer()
        self.object_counter = 0
        self.image_layer = None
        self.progress_bar = None
        self.registration_progress_bar = None
        self.original_image_data = None  # Store original image before registration
        self.is_registered = False  # Track if image has been registered
        
        # Disable built-in layer buttons
        self.viewer.window._qt_viewer.layerButtons.setVisible(False)
        
        # Add custom buttons to the viewer
        self._setup_ui()
        
        # Connect layer events
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.layers.selection.events.active.connect(self._on_layer_selection_changed)
        
    def _setup_ui(self):
        """Setup custom UI buttons"""
        from qtpy.QtWidgets import QProgressBar, QVBoxLayout, QWidget, QLabel, QPushButton
        
        # Create a single registration widget containing all registration controls
        registration_widget = QWidget()
        registration_layout = QVBoxLayout()
        registration_layout.setSpacing(5)
        
        # Register button
        register_btn = QPushButton("Register Image")
        register_btn.clicked.connect(self._register_image)
        registration_layout.addWidget(register_btn)
        
        # Revert button
        revert_btn = QPushButton("Revert Registration")
        revert_btn.clicked.connect(self._revert_registration)
        registration_layout.addWidget(revert_btn)
        
        # Registration progress bar
        self.registration_progress_bar = QProgressBar()
        self.registration_progress_bar.setVisible(False)
        self.registration_progress_bar.setMinimum(0)
        self.registration_progress_bar.setMaximum(100)
        self.registration_progress_bar.setMaximumHeight(20)
        registration_layout.addWidget(self.registration_progress_bar)
        
        registration_widget.setLayout(registration_layout)
        self.viewer.window.add_dock_widget(registration_widget, area='right', name='Image Registration')
        
        # Separator
        separator_widget = QWidget()
        separator_layout = QVBoxLayout()
        separator_layout.setContentsMargins(5, 10, 5, 10)
        separator = QLabel("─" * 30)
        separator.setStyleSheet("color: gray;")
        separator_layout.addWidget(separator)
        separator_widget.setLayout(separator_layout)
        separator_widget.setMaximumHeight(50)
        # self.viewer.window.add_dock_widget(separator_widget, area='right', name='-')
        
        # Create a single segmentation widget containing all segmentation controls
        segmentation_widget = QWidget()
        segmentation_layout = QVBoxLayout()
        segmentation_layout.setSpacing(5)
        
        # Add Object Layer button
        add_object_btn = QPushButton("Add Object Layer")
        add_object_btn.clicked.connect(self._add_object_layer)
        segmentation_layout.addWidget(add_object_btn)
        
        # Plus point mode button
        plus_btn = QPushButton("Plus Point Mode (+)")
        plus_btn.clicked.connect(self._set_plus_mode)
        segmentation_layout.addWidget(plus_btn)
        
        # Minus point mode button
        minus_btn = QPushButton("Minus Point Mode (-)")
        minus_btn.clicked.connect(self._set_minus_mode)
        segmentation_layout.addWidget(minus_btn)
        
        # Segment button
        segment_btn = QPushButton("Segment")
        segment_btn.clicked.connect(self._run_segmentation)
        segmentation_layout.addWidget(segment_btn)
        
        # Segmentation progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMaximumHeight(20)
        segmentation_layout.addWidget(self.progress_bar)
        
        segmentation_widget.setLayout(segmentation_layout)
        self.viewer.window.add_dock_widget(segmentation_widget, area='right', name='Object Segmentation')
        
    def _add_object_layer(self):
        """Add a new object layer with plus/minus points"""
        if self.image_layer is None:
            napari.utils.notifications.show_warning("Please load a 3D image first!")
            return
            
        self.object_counter += 1
        layer_name = f"Object_{self.object_counter}"
        
        # Create points layer - we'll manage symbols manually via properties
        points_layer = self.viewer.add_points(
            ndim=3,
            name=layer_name,
            size=20,  # Larger size to make symbols visible
            properties={
                'point_type': [],      # Track point type (+1 or -1)
                'point_symbol': [],    # Track symbol per point ('cross' or 'hbar')
                'point_color': []      # Track color per point ('green' or 'red')
            },
        )
        
        # Set initial state for plus points (cross symbol)
        points_layer.current_properties = {
            'point_type': 1,
            'point_symbol': 'cross',
            'point_color': 'green'
        }
        points_layer.border_color = 'white'
        points_layer.border_width = 0.2
        points_layer.mode = 'add'
        
        # Initial appearance - set defaults
        self._update_all_points_appearance(points_layer)
        
        # Connect event to update visual appearance when points are added
        points_layer.events.data.connect(lambda e: self._update_all_points_appearance(points_layer))
        
        # Disable layer controls that shouldn't be accessible
        self._disable_layer_controls(points_layer)
        
        napari.utils.notifications.show_info(
            f"Added {layer_name}. Use 'Plus Point Mode' or 'Minus Point Mode' buttons to switch."
        )
    
    def _update_all_points_appearance(self, points_layer):
        """Update the visual appearance of ALL points based on their properties"""
        if len(points_layer.data) == 0:
            return
        
        try:
            # Get the properties
            symbols = points_layer.properties.get('point_symbol', [])
            colors = points_layer.properties.get('point_color', [])
            
            if len(symbols) > 0 and len(colors) > 0:
                # Convert to numpy arrays for napari
                symbols_array = np.array(symbols, dtype=object)
                
                # Convert color strings to RGB arrays
                color_map = {
                    'green': np.array([0, 1, 0, 1]),  # RGBA for green
                    'red': np.array([1, 0, 0, 1])     # RGBA for red
                }
                colors_rgba = np.array([color_map.get(c, [1, 1, 1, 1]) for c in colors])
                
                # Set the symbols and colors for all points
                points_layer.symbol = symbols_array
                points_layer.face_color = colors_rgba
                
        except Exception as e:
            print(f"Error updating points appearance: {e}")
    
    def _disable_layer_controls(self, points_layer):
        """Disable/hide unnecessary layer controls"""
        # Get the Qt layer controls widget
        try:
            from qtpy.QtWidgets import QComboBox, QPushButton, QLabel, QSlider, QCheckBox
            
            qt_viewer = self.viewer.window._qt_viewer
            layer_controls = qt_viewer.controls.widgets[points_layer]
            
            # List of control labels/names to hide
            controls_to_hide = [
                'opacity', 'blending', 'symbol', 'size', 'face_color', 
                'border_color', 'border_width', 'out_of_slice', 'projection_mode',
                'n_dimensional', 'shading', 'antialiasing', 'canvas_size_limits'
            ]
            
            # Recursively find and hide specific control widgets
            def hide_controls(widget, parent_label=None):
                for i, child in enumerate(widget.children()):
                    # Get label text if this is a label
                    current_label = None
                    if isinstance(child, QLabel):
                        current_label = child.text().lower().replace(':', '').replace(' ', '_')
                    
                    # Check if we should hide this control
                    should_hide = False
                    
                    # Hide based on label
                    if current_label and any(name in current_label for name in controls_to_hide):
                        should_hide = True
                    
                    # Hide symbol combo boxes
                    if isinstance(child, QComboBox):
                        should_hide = True
                    
                    # Hide color selector buttons (they usually have color in their style)
                    if isinstance(child, QPushButton):
                        if hasattr(child, 'toolTip'):
                            tooltip = child.toolTip().lower()
                            if any(word in tooltip for word in ['color', 'face', 'border']):
                                should_hide = True
                    
                    # Hide sliders for opacity, size, etc
                    if isinstance(child, QSlider):
                        should_hide = True
                    
                    # Hide checkboxes for out of slice, etc
                    if isinstance(child, QCheckBox):
                        should_hide = True
                    
                    # Apply hiding
                    if should_hide:
                        child.setVisible(False)
                        child.setEnabled(False)
                        # Also hide the next widget (usually the control itself)
                        if i + 1 < len(widget.children()):
                            next_child = widget.children()[i + 1]
                            next_child.setVisible(False)
                            next_child.setEnabled(False)
                    
                    # Recursively check children
                    if hasattr(child, 'children'):
                        hide_controls(child, current_label)
            
            hide_controls(layer_controls)
            
        except (AttributeError, KeyError, ImportError) as e:
            # If we can't access Qt controls, just pass
            print(f"Could not disable layer controls: {e}")
    
    def _set_plus_mode(self):
        """Set active points layer to plus mode"""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None or not isinstance(active_layer, Points):
            napari.utils.notifications.show_warning("Please select a points layer first!")
            return
        
        # Deselect all points to prevent modifying selected ones
        active_layer.selected_data = set()
        
        # Only change the properties for NEW points, don't modify existing ones
        active_layer.current_properties = {
            'point_type': 1,
            'point_symbol': 'cross',
            'point_color': 'green'
        }
        active_layer.mode = 'add'
        napari.utils.notifications.show_info(f"Plus mode activated for {active_layer.name}")
    
    def _set_minus_mode(self):
        """Set active points layer to minus mode"""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None or not isinstance(active_layer, Points):
            napari.utils.notifications.show_warning("Please select a points layer first!")
            return
        
        # Deselect all points to prevent modifying selected ones
        active_layer.selected_data = set()
        
        # Only change the properties for NEW points, don't modify existing ones
        active_layer.current_properties = {
            'point_type': -1,
            'point_symbol': 'hbar',
            'point_color': 'red'
        }
        active_layer.mode = 'add'
        napari.utils.notifications.show_info(f"Minus mode activated for {active_layer.name}")
    
    def _on_layer_inserted(self, event):
        """Handle new layer insertion"""
        layer = event.value
        
        # Check if it's an image layer
        if isinstance(layer, Image):
            if self.image_layer is not None and layer != self.image_layer:
                # Remove the new image layer if one already exists
                napari.utils.notifications.show_warning("Only one image layer allowed!")
                self.viewer.layers.remove(layer)
            else:
                self.image_layer = layer
                napari.utils.notifications.show_info("3D image loaded. Use 'Add Object Layer' to start annotating.")
    
    def _on_layer_removed(self, event):
        """Handle layer removal"""
        layer = event.value
        if isinstance(layer, Image) and layer == self.image_layer:
            self.image_layer = None
            self.original_image_data = None
            self.is_registered = False
    
    def _on_layer_selection_changed(self, event):
        """Handle layer selection changes - hide controls for points layers"""
        active_layer = event.value
        if isinstance(active_layer, Points):
            self._disable_layer_controls(active_layer)
    
    def _run_segmentation(self):
        """Run segmentation based on annotated points"""
        if self.image_layer is None:
            napari.utils.notifications.show_warning("No image loaded!")
            return
        
        # Always convert to numpy array (handles napari Array, Dask, etc)
        image_data = self.image_layer.data
        if not isinstance(image_data, np.ndarray):
            image_data = np.asarray(image_data)
        
        # Iterate through all points layers
        points_layers = [layer for layer in self.viewer.layers if isinstance(layer, Points)]
        
        if not points_layers:
            napari.utils.notifications.show_warning("No object layers found!")
            return
        
        # Show and reset progress bar
        if self.progress_bar:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.viewer.window._qt_window.repaint()  # Force GUI update
        
        # Build list of ClicksInfo objects from all points layers
        clicks_list = []
        layer_names = []
        
        for points_layer in points_layers:
            if len(points_layer.data) == 0:
                continue
            
            # Get points and their properties
            points = points_layer.data
            properties = points_layer.properties
            
            if 'point_type' not in properties or len(properties['point_type']) == 0:
                continue
            
            # Separate positive and negative points
            point_types = np.array(properties['point_type'])
            
            positive_points = []
            negative_points = []
            
            for i, (point, point_type) in enumerate(zip(points, point_types)):
                z, y, x = point
                if point_type == 1:  # Plus point
                    positive_points.append(Point(x=x, y=y, z=z))
                elif point_type == -1:  # Minus point
                    negative_points.append(Point(x=x, y=y, z=z))
            
            # Only add if there are any points
            if positive_points or negative_points:
                clicks_info = ClicksInfo(positive=positive_points, negative=negative_points)
                clicks_list.append(clicks_info)
                layer_names.append(points_layer.name)
        
        if not clicks_list:
            napari.utils.notifications.show_warning("No valid points found in object layers!")
            if self.progress_bar:
                self.progress_bar.setVisible(False)
            return
        
        # Update progress
        if self.progress_bar:
            self.progress_bar.setValue(20)
            self.viewer.window._qt_window.repaint()
        
        try:
            # Run actual segmentation
            napari.utils.notifications.show_info(f"Running segmentation on {len(clicks_list)} object(s)...")
            print("-----------")
            print(image_data.shape, clicks_list)

            segmentations = infer_3d(image_data, clicks_list)
            
            # Update progress
            if self.progress_bar:
                self.progress_bar.setValue(80)
                self.viewer.window._qt_window.repaint()
            
            # Add segmentation results as labels layers
            for idx, (segmentation, layer_name) in enumerate(zip(segmentations, layer_names)):
                seg_layer_name = f"{layer_name}_segmentation"
                
                # Remove old segmentation layer if exists
                existing_seg = [layer for layer in self.viewer.layers if layer.name == seg_layer_name]
                for layer in existing_seg:
                    self.viewer.layers.remove(layer)
                
                # Add new segmentation
                self.viewer.add_labels(segmentation, name=seg_layer_name, opacity=0.5)
            
            # Update progress to complete
            if self.progress_bar:
                self.progress_bar.setValue(100)
                self.viewer.window._qt_window.repaint()
            
            napari.utils.notifications.show_info(f"Created {len(segmentations)} segmentation(s)!")
            
        except Exception as e:
            napari.utils.notifications.show_error(f"Segmentation failed: {str(e)}")
            print(f"Segmentation error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Hide progress bar after completion
            if self.progress_bar:
                self.progress_bar.setVisible(False)
    
    def _register_image(self):
        """Register (align) all slices of the 3D image using rigid 2D transforms"""
        if self.image_layer is None:
            napari.utils.notifications.show_warning("No image loaded!")
            return
        
        # Check if image is 3D
        if self.image_layer.data.ndim != 3:
            napari.utils.notifications.show_warning("Registration only works with 3D images!")
            return
        
        # Check if image has enough slices
        if self.image_layer.data.shape[0] < 2:
            napari.utils.notifications.show_warning("Image must have at least 2 slices for registration!")
            return
        
        if self.is_registered:
            napari.utils.notifications.show_warning("Image is already registered! Use 'Revert' first if you want to re-register.")
            return
        
        # Store original image data
        self.original_image_data = self.image_layer.data.copy()
        
        # Show progress bar
        if self.registration_progress_bar:
            self.registration_progress_bar.setVisible(True)
            self.registration_progress_bar.setValue(0)
            self.viewer.window._qt_window.repaint()
        
        try:
            napari.utils.notifications.show_info("Starting image registration...")
            
            volume = self.image_layer.data
            Z = volume.shape[0]
            
            # Start from middle slice
            middle_idx = Z // 2
            
            corrected_slices = [None] * Z
            corrected_slices[middle_idx] = volume[middle_idx]  # Keep middle slice as is (no registration)
            
            total_iterations = Z - 1  # All slices except the middle
            completed = 0
            
            # Register slices forward from middle (each slice to previous slice)
            for i in range(middle_idx + 1, Z):
                reference_slice = sitk.GetImageFromArray(corrected_slices[i - 1].astype(np.float32))
                corrected_slices[i] = self._register_slice(reference_slice, volume[i])
                completed += 1
                if self.registration_progress_bar:
                    progress = int((completed / total_iterations) * 100)
                    self.registration_progress_bar.setValue(progress)
                    self.viewer.window._qt_window.repaint()
            
            # Register slices backward from middle (each slice to previous slice in sequence)
            for i in range(middle_idx - 1, -1, -1):
                reference_slice = sitk.GetImageFromArray(corrected_slices[i + 1].astype(np.float32))
                corrected_slices[i] = self._register_slice(reference_slice, volume[i])
                completed += 1
                if self.registration_progress_bar:
                    progress = int((completed / total_iterations) * 100)
                    self.registration_progress_bar.setValue(progress)
                    self.viewer.window._qt_window.repaint()
            
            # Stack corrected slices
            corrected_volume = np.stack(corrected_slices, axis=0)
            
            # Update the image layer with registered data
            self.image_layer.data = corrected_volume
            self.is_registered = True
            
            # Complete progress
            if self.registration_progress_bar:
                self.registration_progress_bar.setValue(100)
                self.viewer.window._qt_window.repaint()
            
            napari.utils.notifications.show_info("Image registration completed successfully!")
            
        except Exception as e:
            napari.utils.notifications.show_error(f"Registration failed: {str(e)}")
            print(f"Registration error: {e}")
            import traceback
            traceback.print_exc()
            
            # Revert to original if registration failed
            if self.original_image_data is not None:
                self.image_layer.data = self.original_image_data
                self.original_image_data = None
            
        finally:
            # Hide progress bar
            if self.registration_progress_bar:
                self.registration_progress_bar.setVisible(False)
    
    def _register_slice(self, reference_slice, moving_slice_array):
        """Register a single slice to the reference using rigid 2D transform"""
        moving_slice = sitk.GetImageFromArray(moving_slice_array.astype(np.float32))
        
        # Rigid 2D transform (rotation + translation)
        initial_transform = sitk.CenteredTransformInitializer(
            reference_slice,
            moving_slice,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=100
        )
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # Suppress optimizer output
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        final_transform = registration_method.Execute(reference_slice, moving_slice)
        
        # Apply the transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_slice)
        resampler.SetTransform(final_transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        corrected_slice = resampler.Execute(moving_slice)
        
        return sitk.GetArrayFromImage(corrected_slice)
    
    def _revert_registration(self):
        """Revert the image to its original state before registration"""
        if self.image_layer is None:
            napari.utils.notifications.show_warning("No image loaded!")
            return
        
        if not self.is_registered:
            napari.utils.notifications.show_warning("Image has not been registered yet!")
            return
        
        if self.original_image_data is None:
            napari.utils.notifications.show_error("Original image data not found!")
            return
        
        # Restore original image
        self.image_layer.data = self.original_image_data
        self.original_image_data = None
        self.is_registered = False
        
        napari.utils.notifications.show_info("Image reverted to original state!")
    
    def run(self):
        """Start the napari viewer"""
        napari.run()


def main():
    app = SegmentationApp()
    app.run()


if __name__ == "__main__":
    main()
