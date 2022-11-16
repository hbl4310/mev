This file contains Matlab software that executes the maximum entropy
voting system (as defined (MEV0) in Sewell, MacKay, and McLean
(forthcoming)).

The software is the bare bones of what is needed to run these
systems. There is no attempt made at a user-friendly interface, GUI,
etc.; it is intended for users that are interested in testing the
MEV0 system and who are prepared to read the instructions etc., and if
necessary the code itself, to work out what is going on. 

Installation:
------------

If you are reading this then the first stage of installation has
probably worked.

To test that the Matlab software is running correctly in your version
of Matlab, run the commands

sim1(5, 6); % should produce the output in the file
            % testoutputfromsim1.txt; 
sim2(5, 200, 6); % should produce the output in testoutputfromsim2.txt 

On a PC running Windows NT it is also possible to run a version which
does not depend on Matlab. In order for either of these to run it is
necessary that all the Matlab run-time library files (.dll files) are
available on the appropriate path.

The self-expanding archive mglinstaller.exe needs to be run. It will
request a directory name in which to install the libraries. This
directory must then be added to the appropriate path.

On the PC the appropriate command is

set path=C:\installdirname\bin\win32;%PATH%

where C:\installdirname is the name you gave to mglinstaller. Note
that the PC version works from the command line in the DOS prompt
window, NOT as a Windows application, and that after starting the DOS
prompt window the above path-setting command must be run (each time).

Usage:
-----

Type e.g.

help sim2

within Matlab to get the help instructions printed out, or 

mainsim2

from the shell prompt (or command line window) for the
Matlab-independent versions (which then proceed to run a default
problem, which can be escaped from with control-C).

There are three versions of the software. sim1.m and sim2.m require
Matlab to run. mainsim2 (resp. mainsim2.exe) will run without an
installed copy of Matlab.

sim2.m runs the Markov chain Monte-Carlo version of the
algorithm. This version runs a number of iterations of a sampling
algorithm, whose distribution gradually converges to the output
distribution of the system. How long it takes depends on the settings
of Niters - the larger this variable is set, the longer it takes, and
the more accurately the result distribution will be to the desired
distribution. Niters=1000 seems a reasonable compromise.

sim1.m runs the direct approach. This approach will not be practical
for numbers of candidates in excess of about 6. It is very slow for
more than 3 or 4 candidates. Moreover there are certain circumstances
where the voting pattern leads to the algorithm getting
stuck. Therefore this version is only recommended for very small
numbers of candidates. Resolution of the occasional bugs is not
planned as it is generally recommended to use sim2.

mainsim2 (resp. mainsim2.exe) is a Solaris (resp. PC NT) binary
command-line only version of sim2.m that uses the Matlab run-time
library. If you want to use this version then at the time of calling
the directory containing all the library files must be on your path. This
version has been tested using an NT system; no support whatever will
be provided!

All versions of the software will synthesise their own problems at
random to work on if desired; instructions for invoking this option
are in the help instructions. Both the testing commands above
synthesise problems before working on them.

