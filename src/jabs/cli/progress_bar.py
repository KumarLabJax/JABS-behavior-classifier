

def cli_progress_bar(completed: int, total_iterations: int, length=40,
                     fill_char='█', padding_char='░', prefix='', suffix='',
                     precision=1, complete_as_percent=True):
    """Call in a loop to create terminal progress bar. The loop can't print
    any other output to stdout.

    Args:
        completed: number of completed iterations
        total_iterations: total number of iterations in calling loop
        length: length of bar in characters
        fill_char: filled bar character
        padding_char: unfilled bar character
        prefix: prefix string, printed before progress bar
        suffix: suffix string, printed after progress bar
        precision: number of decimals in percent complete
        complete_as_percent: if True, print percent complete, if False print "num_complete of total_num"
    """

    if len(fill_char) != 1:
        raise ValueError("Invalid fill character")
    if len(padding_char) != 1:
        raise ValueError("Invalid padding character")
    if precision < 1:
        raise ValueError("Invalid precision parameter")
    if length < 1:
        raise ValueError("Invalid length parameter")

    if total_iterations > 0:
        # create a string representation of the percent complete
        complete = f"{100 * (completed / float(total_iterations)):.{precision}f}"

        # calculate the length of the filled portion of the progress bar
        filled_length = int(length * completed // total_iterations)
    else:
        complete = 0
        filled_length = 0

    # create a string combining filled and unfilled portions
    bar = fill_char * filled_length + padding_char * (length - filled_length)

    if complete_as_percent:
        complete = f"{complete}%"
    else:
        width = len(str(total_iterations))
        complete = f"{completed:{width}} of {total_iterations}"

    # print progress bar, overwriting the current line
    print(f'\r{prefix}{bar} {complete} {suffix}', end='\r')

    # print newline once the progress bar is filled
    if completed == total_iterations:
        print()
