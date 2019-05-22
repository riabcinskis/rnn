%
% All things related formatting goes here
%

\ProvidesClass{ktu_phd_summary}[2018/07/03 KTU PhD thesis summary class]
\NeedsTeXFormat{LaTeX2e}
\LoadClass[12pt, a5paper]{article}

%%
%% System
%%
\RequirePackage{afterpackage} % for configuring loaded packages

\RequirePackage{setspace}

\AfterPackage{setspace}{
  % \onehalfspacing
  \linespread{1.15}
  %\doublespacing
}

\RequirePackage{xcolor}

\RequirePackage{geometry} % for page format
\RequirePackage{fancyhdr} % for header, footer formating
\RequirePackage{titlesec} % for section formating



\RequirePackage{ktu_phd} % for thesis document behaviour

\usepackage[lithuanian,german]{babel}

%%
%% Fonts
%%
\AfterPackage{fontspec}{
  \setmainfont{TimesNewRoman.ttf}[
    BoldFont = TimesNewRomanBold.ttf,
    ItalicFont     = TimesNewRomanItalic.ttf,
    BoldItalicFont = TimesNewRomanBoldItalic.ttf]

  \newcommand{\fontIX}{\fontsize{9}{10.8}\selectfont}
  \newcommand{\fontX}{\fontsize{10}{12}\selectfont}
  \newcommand{\fontXI}{\fontsize{11}{13.2}\selectfont}
  \newcommand{\fontXII}{\fontsize{12}{14.4}\selectfont}
  \newcommand{\fontXIII}{\fontsize{13}{15.6}\selectfont}
  \newcommand{\fontXIV}{\fontsize{14}{16.8}\selectfont}
  \newcommand{\fontXV}{\fontsize{15}{18}\selectfont}


  \newcommand{\fontXVIII}{\fontsize{18}{21.6}\selectfont}
}

\RequirePackage{fontspec}

%%
%% English
%%
\AfterPackage{polyglossia}{\setmainlanguage[variant=uk]{english}}
\RequirePackage{polyglossia}

%%
%% Algorithms
%%

% \AfterPackage{algorithm}{
%
%   \algsetup{linenodelimiter=.}
% }
%
% \RequirePackage{algorithm}
%  \RequirePackage{algorithmic}
%  \RequirePackage{algpseudocode}



%%
%% Paragraph indentation
%%
\AfterPackage{parskip}{\setlength\parindent{0.0cm}}
\RequirePackage{parskip}

%%
%% Figure, table caption format
%%


\AfterPackage{caption}{

\DeclareCaptionFormat{figureFormat}{#1#3}
\DeclareCaptionLabelFormat{figureLabelFormat}{#2 #1}

\DeclareCaptionFormat{tableFormat}{#1#3}
\DeclareCaptionLabelFormat{tableLabelFormat}{#2 #1}

\DeclareCaptionFormat{algorithmFormat}{#1#3#2}
\DeclareCaptionLabelFormat{algorithmLabelFormat}{#2 #1}


  \captionsetup[figure]{
    font={small},
     %labelfont={bf},
     format={figureFormat},
     labelformat={figureLabelFormat},
     labelsep=period,
     name={pav. }
  }

  \captionsetup[table]{
    font={small},
     singlelinecheck={false},
     labelfont={bf},
     format={tableFormat},
     labelformat={tableLabelFormat},
     labelsep=period,
     name={lentelė. }
  }

%\DeclareCaptionStyle{ruled}%{labelfont=bf,labelsep=space,strut=off}

  \captionsetup[algorithm]{
     font={small},
    %  labelfont={bf},
    %  format={figureFormat},
    %  labelformat={figureLabelFormat},
    %  labelsep=period,
    format={algorithmFormat},
    labelformat={algorithmLabelFormat},
    labelfont={bf},
    name={algoritmas.\enskip},
    labelsep={space}
  }
}
\RequirePackage{caption} % for figure, table, etc. caption formating

\AfterPackage{chngcntr}{
%  \counterwithin{figure}{section}
%  \counterwithin{table}{section}
}
\RequirePackage{chngcntr}


%\counterwithout{figure}{section}

%%
%% Table of contents, lists of figures/tables
%%

\AfterPackage{tocloft}{

   \renewcommand{\@pnumwidth}{12pt}
   \renewcommand{\@tocrmarg}{36pt}
   \renewcommand{\cftdotsep}{2}

   % table of contents - title
   \setlength\cftbeforetoctitleskip{6pt}
   \setlength\cftaftertoctitleskip{6pt}
   \renewcommand{\cfttoctitlefont}{\hfill\normalfont\bfseries\MakeUppercase}
   \renewcommand{\cftaftertoctitle}{\hfill}
   \addto\captionsenglish{%
     \renewcommand{\contentsname}{Turinys}%
   }

   % table of contents - entries

   \setlength\cftsecindent{0pt}
   \setlength\cftsubsecindent{12pt}
   \setlength\cftsubsubsecindent{24pt}

   \renewcommand{\cftsecfont}{\normalfont}
   \renewcommand{\cftsubsecfont}{\normalfont}
   \renewcommand{\cftsubsubsecfont}{\normalfont}

   \renewcommand{\cftsecpagefont}{\normalfont}
   \renewcommand{\cftsubsecpagefont}{\normalfont}
   \renewcommand{\cftsubsubsecpagefont}{\normalfont}

   \renewcommand{\cftsecaftersnum}{.}
   \renewcommand{\cftsubsecaftersnum}{.}
   \renewcommand{\cftsubsubsecaftersnum}{.}

   \setlength\cftbeforesecskip{0pt}
   \setlength\cftbeforesubsecskip{0pt}
   \setlength\cftbeforesubsubsecskip{0pt}

   \setlength\cftsecnumwidth{12pt}
   \setlength\cftsubsecnumwidth{22pt}
   \setlength\cftsubsubsecnumwidth{30pt}

   \renewcommand{\cftsecdotsep}{\cftdotsep}
   \renewcommand{\cftsubsecdotsep}{\cftdotsep}
   \renewcommand{\cftsubsubsecdotsep}{\cftdotsep}

   \renewcommand{\cftsecleader}{\normalfont\cftdotfill{\cftsecdotsep}}
   \renewcommand{\cftsubsecleader}{\normalfont\cftdotfill{\cftsubsecdotsep}}
   \renewcommand{\cftsubsubsecleader}{\normalfont\cftdotfill{\cftsubsubsecdotsep}}

  % list of figures

   \setlength\cftfigindent{0pt}
   \renewcommand{\cftfigaftersnum}{.}

   \setlength\cftbeforeloftitleskip{6pt}
   \setlength\cftafterloftitleskip{6pt}
   \renewcommand{\cftloftitlefont}{\hfill\normalfont\bfseries\MakeUppercase}
   \renewcommand{\cftafterloftitle}{\hfill}
   \addto\captionsenglish{%
      \renewcommand{\listfigurename}{List of figures}%
   }

   % list of tables

   \setlength\cfttabindent{0pt}
   \renewcommand{\cfttabaftersnum}{.}

   \setlength\cftbeforelottitleskip{6pt}
   \setlength\cftafterlottitleskip{6pt}
   \renewcommand{\cftlottitlefont}{\hfill\normalfont\bfseries\MakeUppercase}
   \renewcommand{\cftafterlottitle}{\hfill}
   \addto\captionsenglish{%
      \renewcommand{\listtablename}{List of tables}%
   }
}
\RequirePackage{tocloft}

%%
%% Citing format
%%
\AfterPackage{natbib}{

  \setcitestyle{notesep={, }, round, citesep={m}}
  \renewcommand\bibnumfmt[1]{#1.}
}
\PassOptionsToPackage{numbers}{natbib}
\RequirePackage{natbib}

\RequirePackage{enumitem}


%%
%% Document geometry
%%

\geometry{
   includehead={false},
   includefoot={true},
   layouthoffset={0mm},
   layoutvoffset={0mm},
   %b5paper,  % gives 176mm x 250mm
   paperwidth={210mm},
   paperheight={297mm},
   %twoside,
   left={1.18in},
   right={0.39in},%20 top/bot, 30left, 10right
   top={0.79in},
   bottom={0.5in},
   footskip={0.25in},
}

%%
%% Header/footer format
%%

\fancypagestyle{plain}{%
   \fancyhf{} % clear all header and footer fields
   \fancyfoot[RO,LE]{\thepage}
   \renewcommand{\headrulewidth}{0pt}
   \renewcommand{\footrulewidth}{0pt}
}

\pagestyle{plain}

%%
%% Section/subsection/... format
%%

\newcommand{\Capitalize}[1]{%
   \edef\@tempa{\expandafter\@gobble\string#1}%
   \edef\@tempb{\expandafter\@car\@tempa\@nil}%
   \edef\@tempa{\expandafter\@cdr\@tempa\@nil}%
   \uppercase\expandafter{\expandafter\def\expandafter\@tempb\expandafter{\@tempb}}%
   \@namedef{\@tempb\@tempa}{\expandafter\MakeUppercase\expandafter{#1}}
}

%    \bf, \it - make the heading bold or italic,
%    \scshape - small capitals,
%    \small, \normalsize, \large, \Large, \LARGE, \huge, \Huge - set the font size,
%    \rmfamily, \sffamily, \ttfamily - set the font type to serifed, san serifed or typewriter respectively.
%    \raggedright
\titleformat{\section}
   %[hang] % <shape>
   {\bf} % <format>
   {\thetitle.\quad} % <label>
   {0em} % <sep>
   {} % <before-code>
   [] % <after-code>

\titleformat{\subsection}
   %[hang] % <shape>
   {\bf} % <format>
   {\thetitle.\quad} % <label>
   {0em} % <sep>
   {} % <before-code>
   [] % <after-code>

\titleformat{\subsubsection}
   %[hang] % <shape>
   {\bf} % <format>
   {\thetitle.\quad} % <label>
   {0em} % <sep>
   {} % <before-code>
   [] % <after-code>

\titlespacing{\section}
   {0pt} % <left>
   {12pt} % <before-sep>
   {12pt} % <after-sep>
   [0pt] % <right-sep>

\titlespacing{\subsection}
   {0pt} % <left>
   {6pt} % <before-sep>
   {6pt} % <after-sep>
   [0pt] % <right-sep>

\titlespacing{\subsubsection}
   {0pt} % <left>
   {6pt} % <before-sep>
   {6pt} % <after-sep>
   [0pt] % <right-sep>

%%
%% Math
%%
\RequirePackage{amsmath} % fo equations
\RequirePackage{bm} % for bold text in equations

%%
%% Figures
%%
\PassOptionsToPackage{dvips,xetex}{graphicx}

\AfterPackage{graphicx}{
   \graphicspath{{images/}}
}
\RequirePackage{graphicx} % for graphics
\RequirePackage{flafter} % for placing figures, tables somewhere after references


%%
%% Figure, table reference format
%%

\AfterPackage{cleveref}{
   \crefname{figure}{fig.}{fig.}
   \Crefname{figure}{Fig.}{Fig.}
   \crefname{table}{table}{table}
   \Crefname{table}{Table}{Table}
}

\RequirePackage{cleveref} % for reference formatting

%%
%% Theorem format
%%

\AfterPackage{enumitem}{
  \newlist{thmlist}{enumerate}{1}
  \setlist[thmlist]{label=\textup{(\roman{thmlisti})},ref={(\roman{thmlisti})}}
  \renewcommand{\p@thmlisti}{\perh@ps{\theproposition}}
  \protected\def\perh@ps#1#2{\textup{#1#2}}
  \newcommand{\itemrefperh@ps}[2]{\textup{#2}}
  \newcommand{\itemref}[1]{\begingroup\let\perh@ps\itemrefperh@ps\ref{#1}\endgroup}
}

\RequirePackage{enumitem}

\RequirePackage{amssymb}

\PassOptionsToPackage{thmmarks}{ntheorem}
\PassOptionsToPackage{amsmath}{ntheorem}

\AfterPackage{ntheorem}{

  \theoremstyle{change}
  \theoremheaderfont{\normalfont\bfseries}
  \theorembodyfont{\itshape}
  \theoremseparator{.}
  \theorempreskip{12pt} % \topsep
  \theorempostskip{12pt} % \topsep
  \theoremindent0cm
  \theoremnumbering{arabic}
  \theoremsymbol{}

  \newtheorem{definition}{apibrėžimas}
  \newtheorem{proposition}{teiginys}
  \newtheorem{theorem}{teorema}
  \newtheorem{hypothesis}{hipotezė}

  \theorembodyfont{\normalfont}
  \theoremsymbol{\ensuremath{\triangle}}

  \newtheorem{example}{pavyzdys}

  \theoremstyle{nonumberplain}
  \theoremheaderfont{\normalfont\itshape}
  \theorembodyfont{}
  \theoremseparator{.}
  \theoremsymbol{\ensuremath{\square}}

  \newtheorem{proof}{Proof}

}
\RequirePackage{ntheorem}

%%
%% Picture drawing
%%
\AfterPackage{tikz}{
  \usetikzlibrary {positioning}
  \usetikzlibrary{fit}
}

\RequirePackage{tikz}
