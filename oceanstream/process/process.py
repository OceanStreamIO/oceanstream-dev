import asyncio
from concurrent.futures import ThreadPoolExecutor
from oceanstream.echodata import (
    read_file,
    compute_sv_with_encode_mode,
    enrich_sv_dataset,
    interpolate_sv,
    regrid_dataset
)


async def process_file_with_progress(progress, compute_task_id, echodata, encode_mode="power", waveform_mode="CW",
                                     depth_offset=0):
    loop = asyncio.get_running_loop()
    total_steps = 25
    progress_step = 95 / total_steps

    process_task_future = loop.run_in_executor(None, compute_sv_with_encode_mode, echodata, waveform_mode, encode_mode)

    for step in range(total_steps):
        if process_task_future.done():
            break
        await asyncio.sleep(0.1)
        progress.update(compute_task_id, advance=progress_step)

    sv_dataset = await process_task_future
    sv_enriched = enrich_sv_dataset(
        sv_dataset,
        echodata,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
        depth_offset=depth_offset
    )

    # del echodata
    # sv_enriched_downsampled = regrid_dataset(sv_enriched)
    # sv_enriched = sv_enriched_downsampled

    return sv_enriched


def compute_sv(echodata, encode_mode="power", waveform_mode="CW", depth_offset=0):
    sv_dataset = compute_sv_with_encode_mode(echodata, waveform_mode, encode_mode)

    # sv_processed = interpolate_sv(sv_dataset)
    sv_processed = sv_dataset
    sv_enriched = enrich_sv_dataset(
        sv_processed,
        echodata,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
        depth_offset=depth_offset
    )


    # del echodata
    # sv_enriched_downsampled = regrid_dataset(sv_enriched)
    # sv_enriched = sv_enriched_downsampled

    return sv_enriched


async def read_file_with_progress(config_data, progress, read_task, waveform_mode="CW"):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        total_steps = 20
        progress_step = 95 / total_steps

        read_task_future = loop.run_in_executor(pool, read_file, config_data, True)

        for step in range(total_steps):
            if read_task_future.done():
                break
            await asyncio.sleep(0.1)
            progress.update(read_task, advance=progress_step)

        echodata, encode_mode = await read_task_future

    return echodata, encode_mode


