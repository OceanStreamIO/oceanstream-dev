import os

import echopype as ep
from adlfs import AzureBlobFileSystem


def list_zarr_files(path, azfs=None):
    """List all Zarr files in the Azure Blob Storage container along with their metadata."""

    if azfs is None:
        azfs = get_azfs()

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

    zarr_files = []
    for blob in azfs.ls(path, detail=True):
        if blob['type'] == 'directory' and not blob['name'].endswith('.zarr'):
            subdir_files = list_zarr_files(blob['name'], azfs)
            zarr_files.extend(subdir_files)
        elif blob['name'].endswith('.zarr'):
            zarr_files.append({
                'Key': blob['name'],
                'Size': blob['size'] if blob['size'] else 0,
                'LastModified': blob['last_modified'] if 'last_modified' in blob else 0
            })

    return zarr_files


def get_azfs(storage_config=None):
    """Get the Azure Blob Storage filesystem object using the connection string from environment variables."""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    if connection_string:
        azfs = AzureBlobFileSystem(connection_string=connection_string)
        return azfs

    if storage_config and storage_config['storage_type'] == 'azure':
        azfs = AzureBlobFileSystem(**storage_config['storage_options'])
        return azfs

    return None


def open_zarr_store(store_name, azfs=None, chunks=None):
    """Open a Zarr store from Azure Blob Storage."""
    if azfs is None:
        azfs = get_azfs()

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

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



