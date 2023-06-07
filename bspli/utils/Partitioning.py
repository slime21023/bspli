import torch

def random_partitioning(data ,leaf_size: int) -> list:
    """
    data: ndarray, the data points and indice
    leaf_size: int

    randomly sample from the data points, and execute the binary partitioning
    """

    if data.shape[0] <= leaf_size:
        return [data]

    result =  []
    cdata = data[:, :-1]
    

    # take the two data points
    indices = torch.randint(0, cdata.shape[0], size=(2,))
    first_norm = torch.norm(cdata- cdata[indices[0]], dim=(1))
    second_norm = torch.norm(cdata- cdata[indices[1]], dim=(1))

    first_idx = (first_norm < second_norm).nonzero().flatten()
    second_idx = (first_norm >= second_norm).nonzero().flatten()

    if first_idx.shape[0] > leaf_size:
        result = result + random_partitioning(data[first_idx], leaf_size)
    else:
        result.append(data[first_idx])

    if second_idx.shape[0] > leaf_size:
        result = result + random_partitioning(data[second_idx], leaf_size)
    else:
        result.append(data[second_idx])

    return result



def max_partitioning(data, leaf_size: int) -> list:
    """
    data: ndarray, the data points and indice
    leaf_size: int

    use mean of the data points, and execute the binary partitioning
    """

    

    result = []
    cdata = data[:, :-1]
    mean = torch.mean(cdata, 0)

    if data.shape[0] <= leaf_size:
        return [(mean , data)]

    md = torch.norm(cdata - mean.reshape(1, mean.shape[0]), dim=(1))
    first_max = cdata[torch.argmax(md)]
    fd = torch.norm(cdata - first_max, dim=(1))
    second_max = cdata[torch.argmax(fd)]

    first_norm = torch.norm(cdata- first_max, dim=(1))
    second_norm = torch.norm(cdata- second_max, dim=(1))
    first_idx = (first_norm < second_norm).nonzero().flatten()
    second_idx = (first_norm >= second_norm).nonzero().flatten()
    if first_idx.shape[0] > leaf_size:
        result = result + max_partitioning(data[first_idx], leaf_size)
    else:
        result.append((mean, data[first_idx]))

    if second_idx.shape[0] > leaf_size:
        result = result + max_partitioning(data[second_idx], leaf_size)
    else:
        result.append((mean, data[second_idx]))
    
    return result


def get_local_model_labels(data, leaf_size: int) -> tuple:
    result = {
        "means": None,
        "data": None,
        "ids": None
    }

    leaves = max_partitioning(data, leaf_size)
    for idx,item in enumerate(leaves):
        mean, ldata = item

        if result["means"] == None:
            result["means"] = mean
        else:
            result["means"] = torch.vstack((result["means"], mean))

        ids = torch.full(size=(ldata.shape[0], 1), fill_value=idx)
        if result["ids"] == None:
            result["ids"] = ids
        else:
            result["ids"] = torch.vstack((result["ids"], ids))

        if result["data"] == None:
            result["data"] = ldata
        else:
            result["data"] = torch.vstack((result["data"], ldata))

    return (result["means"], result["data"], result["ids"])


def get_global_model_labels(model_list: list) -> torch.Tensor:
    means_table = None
    for idx, item in enumerate(model_list):
        means = item[0]
        ma = torch.tensor(means)
        ids = torch.full(size=(ma.shape[0], 1), fill_value=idx)
        result = torch.hstack((ma, ids))
        if means_table == None:
            means_table = result
        else:
            means_table = torch.vstack((means_table, result))

    return means_table