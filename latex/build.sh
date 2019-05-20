 #!/bin/bash

echo "Building ..."
rm -rf build
mkdir build
cd build
mkdir images
mkdir sections
mkdir pages




cd ..

cp -a ./images/. ./build/images/
cp -a ./sections/. ./build/sections/
cp -a ./pages/. ./build/pages/





cp ./iso.bst build/
cp -a ./fonts/. ./build/

#cp unicode-math.sty build/


cp ./ktu_phd_summary.cls build/

cp ./main.tex build/
cp ./db.bib build/
cp ktu_phd.sty build/

cd build
#lualatex -interaction nonstopmode main
lualatex main

cp main.log main0.log
bibtex main
lualatex main
#cp main.log main1.log
lualatex main

pdftotext main.pdf -enc UTF-8 - | wc -m >> stats.txt
texcount -total -inc main.tex >> stats.txt

#mv main.pdf santrauka.pdf
