#include "scope.h"

int main(){
    Scope& s=Scope::instance();
    for(double x=0; x<10; x+=0.01){
        double y=std::pow(x,0.5);
        if (x < 2) {
            s.w("z",std::sin(x));
        }
        s.w("x", x).w("y",y);
        if (x > 5) {
            s.w("z",std::sin(x));
        }
        s.tick(0.01);
        sleep(0.01);
    }
    return 0;
}