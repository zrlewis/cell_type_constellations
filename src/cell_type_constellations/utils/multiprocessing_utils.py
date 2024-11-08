
def winnow_process_list(
        process_list):
    """
    Loop over a list of processes, popping out any that have
    been completed. Return the winnowed list of processes.
    Parameters
    ----------
    process_list: List[multiprocessing.Process]
    Returns
    -------
    process_list: List[multiprocessing.Process]
    """
    to_pop = []
    for ii in range(len(process_list)-1, -1, -1):
        if process_list[ii].exitcode is not None:
            to_pop.append(ii)
            if process_list[ii].exitcode != 0:
                raise RuntimeError(
                    "One of the processes exited with code "
                    f"{process_list[ii].exitcode}")
    for ii in to_pop:
        process_list.pop(ii)
    return process_list
