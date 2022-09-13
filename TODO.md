

@patrickcleeve2

BUG
- tutorial instructions wrong
    - views not the same as default loaded
    - 
- Gaussian beam not displaying correctly, likely napari interpolation (need to verify)
- Axicon visualisation is not showing correctly, likely due to napari interpolation
- Suppress error/warning messages in stout
- Remove development print statements etc

General
- see if we can reduce tutorial pixelsize, probably overkill, causing oom -> cant numerical issues, what is a better solution?


Sim
- memory leak #37
- Beam refactor [#34](https://github.com/DeMarcoLab/juno/issues/34)

UI
- clear napari notifications
- metadata: make loading metadata loading / saving work from all paths (relative / abs)
- select from table click
- sort results table
- make pathing work regardless of where sim was run... 
- remove tmp dir one Beam Creation close... -> replace persistent zarr array for sim with dask into direct write


