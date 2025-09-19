#!/usr/bin/env python3
"""
Unit tests for the Operation classes (BaseOperation, ImageGenerationOperation, ADetailerOperation).
"""

import unittest
import time
import sys
from unittest.mock import Mock, MagicMock, patch
from multiprocessing.connection import Connection

from PyQt6.QtCore import QObject, QCoreApplication, pyqtSignal
from PyQt6.QtWidgets import QApplication

from operations import BaseOperation, ImageGenerationOperation, ADetailerOperation


class MockOperation(BaseOperation):
    """Mock operation for testing BaseOperation functionality."""

    def create_worker_function(self):
        def mock_worker(conn: Connection):
            """Simple mock worker that echoes messages."""
            try:
                while True:
                    if conn.poll(0.1):
                        message = conn.recv()
                        if message is None:
                            break

                        if message == "test_success":
                            conn.send({
                                'type': 'result',
                                'success': True,
                                'data': {'message': 'success'}
                            })
                        elif message == "test_error":
                            conn.send({
                                'type': 'result',
                                'success': False,
                                'error': 'test error message'
                            })
                        elif message == "test_progress":
                            conn.send({
                                'type': 'progress',
                                'progress': 50,
                                'total': 100,
                                'data': {'status': 'in_progress'}
                            })
                        else:
                            conn.send({
                                'type': 'error',
                                'message': f'unknown message: {message}'
                            })
            except Exception as e:
                conn.send({
                    'type': 'error',
                    'message': f'worker error: {e}'
                })
            finally:
                try:
                    conn.close()
                except:
                    pass

        return mock_worker


class TestBaseOperation(unittest.TestCase):
    """Test cases for BaseOperation class."""

    @classmethod
    def setUpClass(cls):
        """Set up Qt application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        """Set up test fixtures."""
        self.operation = MockOperation()
        self.signals_received = {
            'started': [],
            'finished': [],
            'success': [],
            'error': [],
            'progress': []
        }

        # Connect signals to capture them
        self.operation.started.connect(lambda: self.signals_received['started'].append(True))
        self.operation.finished.connect(lambda: self.signals_received['finished'].append(True))
        self.operation.success.connect(lambda data: self.signals_received['success'].append(data))
        self.operation.error.connect(lambda msg: self.signals_received['error'].append(msg))
        self.operation.progress.connect(lambda p, t, d: self.signals_received['progress'].append((p, t, d)))

    def tearDown(self):
        """Clean up after tests."""
        if self.operation:
            try:
                self.operation.stop()
            except:
                pass

    def test_operation_start_and_stop(self):
        """Test basic start and stop functionality."""
        # Initially not running
        self.assertFalse(self.operation._is_running)

        # Start operation
        success = self.operation.start()
        self.assertTrue(success)
        self.assertTrue(self.operation._is_running)
        self.assertIsNotNone(self.operation._process_manager)

        # Should emit started signal
        self.assertEqual(len(self.signals_received['started']), 1)

        # Stop operation
        self.operation.stop()
        self.assertFalse(self.operation._is_running)

        # Should emit finished signal
        self.assertEqual(len(self.signals_received['finished']), 1)

    def test_operation_already_running(self):
        """Test starting operation when already running."""
        # Start operation
        self.assertTrue(self.operation.start())

        # Try to start again - should return True but not create new process
        self.assertTrue(self.operation.start())

        # Should only emit started signal once
        self.assertEqual(len(self.signals_received['started']), 1)

    def test_send_task_success(self):
        """Test sending task successfully."""
        self.operation.start()

        # Send task
        success = self.operation.send_task("test_success")
        self.assertTrue(success)

        # Check for response
        time.sleep(0.2)
        self.operation.check_messages()

        # Should receive success signal
        self.assertEqual(len(self.signals_received['success']), 1)
        self.assertEqual(self.signals_received['success'][0]['message'], 'success')

    def test_send_task_error(self):
        """Test sending task that results in error."""
        self.operation.start()

        # Send task that will cause error
        success = self.operation.send_task("test_error")
        self.assertTrue(success)

        # Check for response
        time.sleep(0.2)
        self.operation.check_messages()

        # Should receive error signal
        self.assertEqual(len(self.signals_received['error']), 1)
        self.assertEqual(self.signals_received['error'][0], 'test error message')

    def test_progress_updates(self):
        """Test progress update handling."""
        self.operation.start()

        # Send progress task
        success = self.operation.send_task("test_progress")
        self.assertTrue(success)

        # Check for response
        time.sleep(0.2)
        self.operation.check_messages()

        # Should receive progress signal
        self.assertEqual(len(self.signals_received['progress']), 1)
        progress, total, data = self.signals_received['progress'][0]
        self.assertEqual(progress, 50)
        self.assertEqual(total, 100)
        self.assertEqual(data['status'], 'in_progress')

    def test_send_task_not_running(self):
        """Test sending task when operation not running."""
        # Try to send task without starting
        success = self.operation.send_task("test")
        self.assertFalse(success)

    def test_is_alive(self):
        """Test is_alive functionality."""
        # Not alive when not started
        self.assertFalse(self.operation.is_alive())

        # Start operation
        self.operation.start()

        # Should be alive
        self.assertTrue(self.operation.is_alive())

        # Stop operation
        self.operation.stop()
        time.sleep(0.1)

        # Should not be alive
        self.assertFalse(self.operation.is_alive())


class TestImageGenerationOperation(unittest.TestCase):
    """Test cases for ImageGenerationOperation class."""

    @classmethod
    def setUpClass(cls):
        """Set up Qt application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        """Set up test fixtures."""
        self.operation = ImageGenerationOperation()
        self.signals_received = {
            'preview_image': [],
            'generation_complete': []
        }

        # Connect specific signals
        self.operation.preview_image.connect(
            lambda data, step, steps, width, height, gen_time:
            self.signals_received['preview_image'].append((data, step, steps, width, height, gen_time))
        )
        self.operation.generation_complete.connect(
            lambda data: self.signals_received['generation_complete'].append(data)
        )

    def tearDown(self):
        """Clean up after tests."""
        if self.operation:
            try:
                self.operation.stop()
            except:
                pass

    def test_worker_function_creation(self):
        """Test that worker function can be created."""
        worker_func = self.operation.create_worker_function()
        self.assertTrue(callable(worker_func))

    def test_message_handling_preview_image(self):
        """Test handling of preview image messages."""
        # Mock message with preview data
        message = {
            'type': 'progress',
            'progress': 5,
            'total': 10,
            'data': {
                'preview_data': b'fake_image_data',
                'width': 512,
                'height': 512,
                'generation_time': None
            }
        }

        self.operation._handle_message(message)

        # Should emit preview_image signal
        self.assertEqual(len(self.signals_received['preview_image']), 1)
        data, step, steps, width, height, gen_time = self.signals_received['preview_image'][0]
        self.assertEqual(data, b'fake_image_data')
        self.assertEqual(step, 5)
        self.assertEqual(steps, 10)
        self.assertEqual(width, 512)
        self.assertEqual(height, 512)

    def test_message_handling_generation_complete(self):
        """Test handling of generation complete messages."""
        # Mock result message
        message = {
            'type': 'result',
            'success': True,
            'data': {
                'image_data': b'final_image_data',
                'width': 1024,
                'height': 1024
            }
        }

        self.operation._handle_message(message)

        # Should emit generation_complete signal
        self.assertEqual(len(self.signals_received['generation_complete']), 1)
        result_data = self.signals_received['generation_complete'][0]
        self.assertEqual(result_data['width'], 1024)
        self.assertEqual(result_data['height'], 1024)


class TestADetailerOperation(unittest.TestCase):
    """Test cases for ADetailerOperation class."""

    @classmethod
    def setUpClass(cls):
        """Set up Qt application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        """Set up test fixtures."""
        self.operation = ADetailerOperation()
        self.signals_received = {
            'face_detected': [],
            'inpainting_progress': [],
            'adetailer_complete': []
        }

        # Connect specific signals
        self.operation.face_detected.connect(
            lambda x, y, w, h: self.signals_received['face_detected'].append((x, y, w, h))
        )
        self.operation.inpainting_progress.connect(
            lambda current, total: self.signals_received['inpainting_progress'].append((current, total))
        )
        self.operation.adetailer_complete.connect(
            lambda data: self.signals_received['adetailer_complete'].append(data)
        )

    def tearDown(self):
        """Clean up after tests."""
        if self.operation:
            try:
                self.operation.stop()
            except:
                pass

    def test_worker_function_creation(self):
        """Test that worker function can be created."""
        worker_func = self.operation.create_worker_function()
        self.assertTrue(callable(worker_func))

    def test_message_handling_face_detected(self):
        """Test handling of face detection messages."""
        message = {
            'type': 'progress',
            'progress': 1,
            'total': 1,
            'data': {
                'face_detected': True,
                'face_rect': (100, 150, 200, 250)
            }
        }

        self.operation._handle_message(message)

        # Should emit face_detected signal
        self.assertEqual(len(self.signals_received['face_detected']), 1)
        x, y, w, h = self.signals_received['face_detected'][0]
        self.assertEqual((x, y, w, h), (100, 150, 200, 250))

    def test_message_handling_inpainting_progress(self):
        """Test handling of inpainting progress messages."""
        message = {
            'type': 'progress',
            'progress': 15,
            'total': 20,
            'data': {
                'inpainting_step': True
            }
        }

        self.operation._handle_message(message)

        # Should emit inpainting_progress signal
        self.assertEqual(len(self.signals_received['inpainting_progress']), 1)
        current, total = self.signals_received['inpainting_progress'][0]
        self.assertEqual(current, 15)
        self.assertEqual(total, 20)

    def test_message_handling_adetailer_complete(self):
        """Test handling of ADetailer completion messages."""
        message = {
            'type': 'result',
            'success': True,
            'data': {
                'image_data': b'processed_image_data',
                'width': 512,
                'height': 512
            }
        }

        self.operation._handle_message(message)

        # Should emit adetailer_complete signal
        self.assertEqual(len(self.signals_received['adetailer_complete']), 1)
        result_data = self.signals_received['adetailer_complete'][0]
        self.assertEqual(result_data['width'], 512)
        self.assertEqual(result_data['height'], 512)


class TestOperationIntegration(unittest.TestCase):
    """Integration tests for operations."""

    @classmethod
    def setUpClass(cls):
        """Set up Qt application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_multiple_operations(self):
        """Test running multiple operations simultaneously."""
        mock_op1 = MockOperation()
        mock_op2 = MockOperation()

        try:
            # Start both operations
            self.assertTrue(mock_op1.start())
            self.assertTrue(mock_op2.start())

            # Both should be alive
            self.assertTrue(mock_op1.is_alive())
            self.assertTrue(mock_op2.is_alive())

            # Send tasks to both
            self.assertTrue(mock_op1.send_task("test_success"))
            self.assertTrue(mock_op2.send_task("test_progress"))

            # Brief wait for processing
            time.sleep(0.2)

            # Check messages
            mock_op1.check_messages()
            mock_op2.check_messages()

        finally:
            mock_op1.stop()
            mock_op2.stop()

    def test_operation_inheritance(self):
        """Test that operations properly inherit from BaseOperation."""
        image_gen = ImageGenerationOperation()
        adetailer = ADetailerOperation()

        try:
            # Both should be instances of BaseOperation
            self.assertIsInstance(image_gen, BaseOperation)
            self.assertIsInstance(adetailer, BaseOperation)

            # Both should be QObjects
            self.assertIsInstance(image_gen, QObject)
            self.assertIsInstance(adetailer, QObject)

            # Both should have the required methods
            self.assertTrue(hasattr(image_gen, 'start'))
            self.assertTrue(hasattr(image_gen, 'stop'))
            self.assertTrue(hasattr(image_gen, 'send_task'))
            self.assertTrue(hasattr(image_gen, 'check_messages'))

            self.assertTrue(hasattr(adetailer, 'start'))
            self.assertTrue(hasattr(adetailer, 'stop'))
            self.assertTrue(hasattr(adetailer, 'send_task'))
            self.assertTrue(hasattr(adetailer, 'check_messages'))

            # Both should have specific signals
            self.assertTrue(hasattr(image_gen, 'preview_image'))
            self.assertTrue(hasattr(image_gen, 'generation_complete'))

            self.assertTrue(hasattr(adetailer, 'face_detected'))
            self.assertTrue(hasattr(adetailer, 'inpainting_progress'))
            self.assertTrue(hasattr(adetailer, 'adetailer_complete'))

        finally:
            image_gen.stop()
            adetailer.stop()


class TestOperationErrorHandling(unittest.TestCase):
    """Test error handling in operations."""

    @classmethod
    def setUpClass(cls):
        """Set up Qt application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_invalid_message_handling(self):
        """Test handling of invalid messages."""
        operation = MockOperation()

        # Test with non-dict message
        operation._handle_message("not a dict")
        # Should not raise exception

        # Test with dict without type
        operation._handle_message({'data': 'test'})
        # Should not raise exception

        # Test with unknown message type
        operation._handle_message({'type': 'unknown', 'data': 'test'})
        # Should not raise exception

    @patch('operations.ProcessManager')
    def test_start_failure_handling(self, mock_process_manager_class):
        """Test handling of process start failures."""
        # Make ProcessManager constructor raise exception
        mock_process_manager_class.side_effect = Exception("Failed to create process")

        operation = MockOperation()
        error_messages = []
        operation.error.connect(error_messages.append)

        # Should return False and emit error signal
        success = operation.start()
        self.assertFalse(success)
        self.assertEqual(len(error_messages), 1)
        self.assertIn("Failed to start operation", error_messages[0])


if __name__ == '__main__':
    # Create QApplication if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    unittest.main(verbosity=2)
