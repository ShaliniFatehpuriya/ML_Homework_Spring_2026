set title "Least Square Error"
set xlabel "X"
set ylabel "Y"

set grid
set multiplot

set datafile separator ","
plot "data.txt" using 1:2 with points pt 7 ps 1.5 lc rgb "purple" title "Data" \

set datafile separator " "
plot "predictedLSE.txt" using 1:2 with lines lw 2 lc rgb "blue" title "Model"

unset multiplot
pause -1