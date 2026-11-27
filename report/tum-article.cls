%%
% TUM Corporate Design Article
% Michael Riesch (michael.riesch@tum.de)
% Based on the templates from https://www.tum.de/cd
%
% Template for scientific articles, exercise sheets, notes, ...
%
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tum-article}[2018/10/08 TUM Corporate Design Article v0.3]

% process options
\DeclareOption{twocolumn}{\PassOptionsToClass{twocolumn}{scrartcl}}
\DeclareOption{times}{\PassOptionsToPackage{times}{tum-base}}
\DeclareOption{german}{\PassOptionsToPackage{german}{tum-base}}
\DeclareOption{english}{\PassOptionsToPackage{english}{tum-base}}
\ProcessOptions\relax

% load base class
\LoadClass[fontsize=11pt,a4paper,bibtotoc,numbers=noenddot]{scrartcl}

% load TUM base packages
\RequirePackage{tum-base}
\RequirePackage{tum-base-text}

\RequirePackage{placeins}

% custom email + thanks
\let\theemail\empty
\newcommand{\email}[1]{\renewcommand{\theemail}{#1}}

% ORCID inclusion
\DeclareRobustCommand{\orcid}[1]{\href{http://orcid.org/#1}{%
    \includegraphics[height=\fontcharht\font`\B]{tum-resources/images/orcid.pdf}}
}

% authormark
\RequirePackage[misc]{ifsym}
\newcommand{\authormark}[1]{\textsuperscript{\normalfont#1}}

% affiliations
\newcounter{affil}
\let\theaffil\empty
\newcommand{\printaffil}[2]{\authormark{#1}#2}
\newcommand{\affil}[2][]{%
  \ifnum\value{affil}>0{
    \makeatletter
    \g@addto@macro{\theaffil}{\par\printaffil{#1}{#2}}
    \makeatother
  }\else{
    \makeatletter
    \g@addto@macro{\theaffil}{\printaffil{#1}{#2}}
    \makeatother
  }\fi
  \addtocounter{affil}{1}
}

% custom title
\makeatletter
\def\@maketitle{%
  \newpage
  {
  }
  {\raggedright\fontfamily{phv}\LARGE\bfseries\@title\par}
  {\raggedright\fontfamily{phv}\large\@subtitle\par}
  \vskip 1.5em
  {\raggedright\fontfamily{phv}\bfseries\large\textcolor{TUMBlau}{\@author}
    \par}
  {\raggedright\fontfamily{phv}\small\theaffil\par}
  \ifx\theemail\@empty\relax
  \else{%
    \raggedright\fontfamily{phv}\footnotesize\textcolor{TUMBlau}{\Letter}
    \theemail\par
  }
  \fi
  {\fontfamily{phv}\footnotesize\@date\par}
  \vskip 1.5em
  \FloatBarrier
}
\makeatother

% custom abstract
\makeatletter
\renewenvironment{abstract}{%
  \fontfamily{phv}\selectfont
  \setlength{\parindent}{0mm}
  \color{TUMBlau}\textbf{\abstractname}
  \normalfont\color{black}---}{}
\makeatother

\newcommand\therunningtitle{}
\newcommand\titlerunning[1]{\renewcommand\therunningtitle{#1}}
\newcommand\therunningauthor{}
\newcommand\authorrunning[1]{\renewcommand\therunningauthor{#1}}

% custom header/footer
\RequirePackage{fancyhdr}
\RequirePackage{totcount}
\regtotcounter{page}
\fancypagestyle{plain}{
  % clear everything
  \fancyhead{}
  \fancyfoot{}

  % threeliner
  \fancyhead[L]{%
    \ifnum\thepage<2
    \vbox to\tumlogoheight{%
      \fontfamily{phv}\fontsize{9}{11}\selectfont\textcolor{TUMBlau}{%
        \thegroup\\
        \thedepartment\\
        \theuniversity
      }
    }
    \else\fi
  }

  % logo
  \fancyhead[R]{%
    \ifnum\thepage<2
    \vbox to\tumlogoheight{%
      \includegraphics[height=\tumlogoheight]%
                      {tum-resources/images/Universitaet_Logo_RGB.pdf}
    }
    \else\fi
  }

  % page numbers
  \fancyfoot[R]{%
    \ifnum\totvalue{page}>1
    \fontfamily{phv}\selectfont\textbf{\thepage}
    \else\fi
  }
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\pagestyle{plain}
