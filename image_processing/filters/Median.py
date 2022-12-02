import copy


def median_filter(data, filter_size):
    nimg = copy.copy(data)

    for color in range(3):
        data = nimg[:, :, color]
        indexer = filter_size // 2
        window = [
            (i, j)
            for i in range(-indexer, filter_size-indexer)
            for j in range(-indexer, filter_size-indexer)
        ]
        index = len(window) // 2
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = sorted(
                    0 if (
                        min(i+a, j+b) < 0
                        or len(data) <= i+a
                        or len(data[0]) <= j+b
                    ) else data[i+a][j+b]
                    for a, b in window
                )[index]
        nimg[:, :, color] = data
    return nimg