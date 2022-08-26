#!/usr/bin/env sh

make install

# get the gura DEM
wget https://storage.googleapis.com/releases.naturalcapitalproject.org/invest/3.11.0/data/SDR.zip
unzip SDR.zip

# Get the pitfilled DEM from Wang & Liu
WANGLIU_DEM="wangandliu.sdat"
saga_cmd ta_preprocessor 4 -ELEV "SDR/DEM_gura.tif" -FILLED "$WANGLIU_DEM"

for i in {0..3}
do
    DIR="SAGA-area-$i"
    mkdir "$DIR" || echo "dir already exists"
    saga_cmd ta_hydrology 25 \
        -DEM "$WANGLIU_DEM" \
        -LS_FACTOR "$DIR/SAGA_LS_Factor_area_$i.sdat" \
        -UPSLOPE_AREA "$DIR/SAGA_upslope_area_$i.sdat" \
        -UPSLOPE_LENGTH "$DIR/SAGA_upslope_length_$i.sdat" \
        -METHOD_AREA "$i" \
        -METHOD 1  # Method 1 is Desmet & Govers 1996
done

python3 run-gura.py
