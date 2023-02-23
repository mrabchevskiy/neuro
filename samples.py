"""
 Copyright Mykola Rabchevskiy 2023.
 Distributed under the Boost Software License, Version 1.0.
 (See http://www.boost.org/LICENSE_1_0.txt)

_______________________________________________________________________________

`Training` data set generation

"""
import math
import random

random.seed()

N  = 1000               # :number of samples
A  =    2.0
R2 =    8.0 / math.pi

print( 'R = %.3f' % math.sqrt( R2 ) )


out = open( 'samples.csv', 'w' )
i = 0
while i < N:
  u = random.uniform( -A, A )
  v = random.uniform( -A, A )
  y = 1.0 if ( u**2 + v**2 ) > R2 else 0.0
  out.write( '%f\t%f\t%f\n' % ( u, v, y ) )
  i += 1
out.close()

