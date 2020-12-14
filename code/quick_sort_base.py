# in-place quicksort based on the implementation of 
# StackOverflow user "Ant"
# See: https://stackoverflow.com/questions/17773516/in-place-quicksort-in-python
def sub_partition(q, array, start, end, idx_pivot):
    # Note: q is a precedence matrix
    # See utils.py
    if not (start <= idx_pivot <= end):
        raise ValueError('idx pivot must be between start and end')

    array[start], array[idx_pivot] = array[idx_pivot], array[start]
    pivot = array[start]
    i = start + 1
    j = start + 1

    while j <= end:
        if q[j,pivot] <= q[pivot,j]:
            array[j], array[i] = array[i], array[j]
            i += 1
        j += 1

    array[start], array[i - 1] = array[i - 1], array[start]
    return i - 1

def quicksort(q, array, pivotFunc, start=0, end=None):
    # Note: q is a precedence matrix
    # See utils.py
    if end is None:
        end = len(array) - 1

    if end - start < 1:
        return

    idx_pivot = pivotFunc(array, start, end)
    i = sub_partition(q, array, start, end, idx_pivot)
   
    quicksort(q, array, pivotFunc, start, i - 1)
    quicksort(q, array, pivotFunc, i + 1, end)




