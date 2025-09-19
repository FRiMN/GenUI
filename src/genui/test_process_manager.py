#!/usr/bin/env python3
"""
Unit tests for the ProcessManager class.
"""

import unittest
import time
import threading
from multiprocessing.connection import Connection
from process_manager import ProcessManager


class TestProcessManager(unittest.TestCase):
    """Test cases for ProcessManager class."""

    def test_basic_initialization_and_cleanup(self):
        """Test that ProcessManager initializes and cleans up properly."""
        def simple_worker(conn: Connection):
            time.sleep(0.1)
            conn.send("worker_started")
            while True:
                if conn.poll(0.1):
                    msg = conn.recv()
                    if msg is None:
                        break
                    conn.send(f"echo: {msg}")

        pm = ProcessManager(simple_worker)
        self.assertTrue(pm.is_alive())
        self.assertIsNotNone(pm.parent_conn)
        self.assertIsNotNone(pm.process)

        pm.stop()
        time.sleep(0.1)  # Give time for cleanup
        self.assertFalse(pm.is_alive())

    def test_context_manager(self):
        """Test ProcessManager as context manager."""
        def simple_worker(conn: Connection):
            conn.send("started")
            while conn.poll(1.0):
                msg = conn.recv()
                if msg is None:
                    break

        with ProcessManager(simple_worker) as pm:
            self.assertTrue(pm.is_alive())
            # Should receive startup message
            if pm.poll(1.0):
                msg = pm.recv()
                self.assertEqual(msg, "started")

        # Process should be stopped after context exit
        time.sleep(0.1)
        self.assertFalse(pm.is_alive())

    def test_send_and_receive(self):
        """Test sending and receiving data."""
        def echo_worker(conn: Connection):
            while True:
                if conn.poll(0.1):
                    msg = conn.recv()
                    if msg is None:
                        break
                    if msg == "ping":
                        conn.send("pong")
                    else:
                        conn.send(f"echo: {msg}")

        with ProcessManager(echo_worker) as pm:
            # Test ping-pong
            pm.send("ping")
            if pm.poll(1.0):
                response = pm.recv()
                self.assertEqual(response, "pong")

            # Test echo
            pm.send("hello world")
            if pm.poll(1.0):
                response = pm.recv()
                self.assertEqual(response, "echo: hello world")

    def test_poll_functionality(self):
        """Test poll functionality."""
        def delayed_sender(conn: Connection):
            time.sleep(0.2)  # Delay before sending
            conn.send("delayed_message")
            while conn.poll(1.0):
                msg = conn.recv()
                if msg is None:
                    break

        with ProcessManager(delayed_sender) as pm:
            # Should not be ready immediately
            self.assertFalse(pm.poll(0.1))

            # Should be ready after delay
            self.assertTrue(pm.poll(0.5))

            if pm.poll():
                msg = pm.recv()
                self.assertEqual(msg, "delayed_message")

    def test_recv_timeout(self):
        """Test receive timeout functionality."""
        def slow_worker(conn: Connection):
            time.sleep(2.0)  # Long delay
            conn.send("finally_ready")
            while conn.poll(1.0):
                msg = conn.recv()
                if msg is None:
                    break

        with ProcessManager(slow_worker) as pm:
            # Should timeout
            with self.assertRaises(TimeoutError):
                pm.recv(timeout=0.1)

    def test_send_to_closed_process(self):
        """Test sending to a closed process."""
        def quick_worker(conn: Connection):
            conn.send("started")
            # Exit quickly

        pm = ProcessManager(quick_worker)
        time.sleep(0.5)  # Let process finish

        # Should return False when trying to send to closed process
        result = pm.send("test")
        self.assertFalse(result)

        pm.stop()

    def test_multiple_messages(self):
        """Test sending multiple messages rapidly."""
        def counter_worker(conn: Connection):
            counter = 0
            while True:
                if conn.poll(0.1):
                    msg = conn.recv()
                    if msg is None:
                        break
                    if msg == "increment":
                        counter += 1
                        conn.send(counter)

        with ProcessManager(counter_worker) as pm:
            expected_values = []

            # Send multiple increment commands
            for i in range(5):
                pm.send("increment")
                expected_values.append(i + 1)

            # Receive all responses
            received_values = []
            for _ in range(5):
                if pm.poll(1.0):
                    value = pm.recv()
                    received_values.append(value)

            self.assertEqual(received_values, expected_values)

    def test_worker_exception_handling(self):
        """Test handling of exceptions in worker process."""
        def failing_worker(conn: Connection):
            conn.send("before_error")
            raise RuntimeError("Intentional error")

        with ProcessManager(failing_worker) as pm:
            # Should receive message before error
            if pm.poll(1.0):
                msg = pm.recv()
                self.assertEqual(msg, "before_error")

            # Process should die due to exception
            time.sleep(0.5)
            self.assertFalse(pm.is_alive())

    def test_complex_data_structures(self):
        """Test sending complex data structures."""
        def data_processor(conn: Connection):
            while True:
                if conn.poll(0.1):
                    data = conn.recv()
                    if data is None:
                        break

                    if isinstance(data, dict):
                        # Echo back with added info
                        data['processed'] = True
                        data['timestamp'] = time.time()
                        conn.send(data)
                    elif isinstance(data, list):
                        # Return sum
                        conn.send(sum(data))

        with ProcessManager(data_processor) as pm:
            # Test dictionary
            test_dict = {"name": "test", "value": 42}
            pm.send(test_dict)

            if pm.poll(1.0):
                result = pm.recv()
                self.assertIsInstance(result, dict)
                self.assertTrue(result['processed'])
                self.assertEqual(result['name'], "test")
                self.assertEqual(result['value'], 42)
                self.assertIn('timestamp', result)

            # Test list
            test_list = [1, 2, 3, 4, 5]
            pm.send(test_list)

            if pm.poll(1.0):
                result = pm.recv()
                self.assertEqual(result, 15)

    def test_thread_safety(self):
        """Test thread safety of ProcessManager."""
        def thread_safe_worker(conn: Connection):
            counter = 0
            while True:
                if conn.poll(0.01):
                    msg = conn.recv()
                    if msg is None:
                        break
                    counter += 1
                    conn.send(counter)

        with ProcessManager(thread_safe_worker) as pm:
            results = []

            def sender_thread():
                for i in range(10):
                    pm.send(f"msg_{i}")
                    time.sleep(0.01)

            def receiver_thread():
                for _ in range(10):
                    if pm.poll(1.0):
                        result = pm.recv()
                        results.append(result)

            # Start threads
            t1 = threading.Thread(target=sender_thread)
            t2 = threading.Thread(target=receiver_thread)

            t1.start()
            t2.start()

            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

            # Should have received all messages
            self.assertEqual(len(results), 10)
            self.assertEqual(results, list(range(1, 11)))

    def test_graceful_shutdown(self):
        """Test graceful shutdown of process."""
        shutdown_called = threading.Event()

        def graceful_worker(conn: Connection):
            try:
                while True:
                    if conn.poll(0.1):
                        msg = conn.recv()
                        if msg is None:
                            break
            except:
                pass
            finally:
                # This won't work across processes, but demonstrates cleanup
                pass

        pm = ProcessManager(graceful_worker)
        self.assertTrue(pm.is_alive())

        # Should stop gracefully
        pm.stop(timeout=1.0)
        self.assertFalse(pm.is_alive())

    def test_force_kill_on_unresponsive_process(self):
        """Test force killing of unresponsive process."""
        def unresponsive_worker(conn: Connection):
            # Ignore all signals and just loop forever
            while True:
                time.sleep(1.0)

        pm = ProcessManager(unresponsive_worker)
        self.assertTrue(pm.is_alive())

        # Should eventually kill the process
        start_time = time.time()
        pm.stop(timeout=0.1)  # Very short timeout
        stop_time = time.time()

        self.assertFalse(pm.is_alive())
        # Should not take too long (force kill)
        self.assertLess(stop_time - start_time, 10.0)


if __name__ == '__main__':
    unittest.main()
