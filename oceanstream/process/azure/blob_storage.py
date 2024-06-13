import echopype as ep


def list_zarr_files(azfs, path):
    """List all Zarr files in the Azure Blob Storage container along with their metadata."""
    zarr_files = []
    for blob in azfs.ls(path, detail=True):
        if blob['type'] == 'directory' and not blob['name'].endswith('.zarr'):
            subdir_files = list_zarr_files(azfs, blob['name'])
            zarr_files.extend(subdir_files)
        elif blob['name'].endswith('.zarr'):
            zarr_files.append({
                'Key': blob['name'],
                'Size': blob['size'] if blob['size'] else 0,
                'LastModified': blob['last_modified'] if 'last_modified' in blob else 0
            })

    return zarr_files


def open_zarr_store(azfs, store_name, chunks=None):
    """Open a Zarr store from Azure Blob Storage."""
    mapper = azfs.get_mapper(store_name)

    return ep.open_converted(mapper, chunks=chunks)


def _list_zarr_files_extended(azfs, path):
    """List all Zarr files in the Azure Blob Storage container along with their metadata."""
    zarr_files = []
    for blob in azfs.ls(path, detail=True):
        if blob['type'] == 'directory' and not blob['name'].endswith('.zarr'):
            subdir_files = list_zarr_files(azfs, blob['name'])
            zarr_files.extend(subdir_files)
        else:
            # Calculate the total size and most recent modification date for the .zarr folder
            total_size = 0
            last_modified = None
            for sub_blob in azfs.ls(blob['name'], detail=True):
                if sub_blob['type'] == 'file':
                    total_size += sub_blob['size']
                    if last_modified is None or sub_blob['last_modified'] > last_modified:
                        last_modified = sub_blob['last_modified']

            zarr_files.append({
                'Key': blob['name'],
                'Size': total_size,
                'LastModified': last_modified
            })

    return zarr_files



