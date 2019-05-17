ECHO Let's build latex document

IF NOT EXIST  "build" MKDIR build
CD build
DEL * /F /Q
MKDIR images
MKDIR sections
MKDIR pages
MKDIR files
CD ..

XCOPY .\files .\build\files\ /s /e /y
XCOPY .\images .\build\images\ /s /e /y

XCOPY .\sections .\build\sections\ /s /e /y
XCOPY .\pages .\build\pages\ /s /e /y

COPY .\iso.bst .\build\

COPY .\fonts\TimesNewRoman.ttf .\build\
COPY .\fonts\TimesNewRomanBold.ttf .\build\
COPY .\fonts\TimesNewRomanBoldItalic.ttf .\build\
COPY .\fonts\TimesNewRomanItalic.ttf .\build\



COPY .\ktu_phd_summary.cls .\build\
COPY .\db.bib .\build\

COPY .\ktu_phd.sty .\build\


COPY .\main.tex .\build\



CD build
lualatex main
cp main.log main0.log
bibtex main
lualatex main
cp main.log main1.log
lualatex main


REN main.pdf santrauka.pdf

CD ..