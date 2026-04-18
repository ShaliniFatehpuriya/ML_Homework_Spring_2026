set term pngcairo size 900,700 enhanced
set output "logistic_regression.png"

set multiplot layout 1,3 title "Logistic Regression Comparison"

set xrange [-3:15]
set yrange [-2:8]
set grid

# -------------------------
# 1. Ground Truth
# -------------------------
set title "Ground Truth"
set ylabel "y"

plot "data.dat" using 1:2:($3==0 ? 0xff0000 : 0x0000ff) \
     with points pt 7 ps 1.5 lc rgb variable title "Data"

# -------------------------
# 2. Gradient Descent
# -------------------------
set title "Gradient Descent"
unset ylabel

plot "data.dat" using 1:2:($3==0 ? 0xff0000 : 0x0000ff) \
     with points pt 7 ps 1.5 lc rgb variable notitle, \

# -------------------------
# 3. Newton Method
# -------------------------
set title "Newton Method"

plot "data.dat" using 1:2:($3==0 ? 0xff0000 : 0x0000ff) \
     with points pt 7 ps 1.5 lc rgb variable notitle, \

unset multiplot
unset output