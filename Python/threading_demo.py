# -*- coding: utf-8 -*-
"""
Demo showcasing use of multiple threads to consume work from an input queue and 
push results into an output queue.
"""

import multiprocessing as mp
import queue
import time
import threading
import typing as ty

STOP_SIGNAL = None  # something that would never be in the in_q for actual work

def square_thing(x: int) -> int:
    """
    This is the logic of what you actually want things to
    do. In this example, I'm just squaring numbers, but
    also sleeping a bit to pretend this takes actual time.
    """
    time.sleep(1)
    return x*x

def thread_target(in_q: queue.Queue, out_q: queue.Queue, worker: ty.Callable) -> None:
    """
    This is what your threads execute. It's job is to get 
    work from the input queue, get the result using
    the specified worker function, and put the result in
    the output queue. It must also know how to tell
    when there is no more work.
    """
    while True:
        work = in_q.get()
        if work == STOP_SIGNAL:
            in_q.put(work)  # so other threads get stop signal
            break
        out_q.put(worker(work))

def main():
    in_q = queue.Queue()
    out_q = queue.Queue()

    # initialize threads
    threads = []
    for _ in range(mp.cpu_count()):
        thread = threading.Thread(
            target=thread_target,
            args=(in_q, out_q, square_thing),
        )
        thread.start()
        threads.append(thread)

    # Put work in in_q
    for i in range(100):
        in_q.put(i)

    # Put stop signal in in_q to indicate no more work
    in_q.put(STOP_SIGNAL)

    # wait for threads to finish
    for thread in threads:
        thread.join()

    # get results
    while not out_q.empty():
        print(out_q.get())