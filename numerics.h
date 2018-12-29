#pragma once

#include <cmath>
#include <stdexcept>

namespace harris {

// Clamps a value to a given range by truncating it to max or min.
float Clamp(float value, float min, float max) {
    if (value > max) return max;
    if (value < min) return min;
    return value;
}

// Clamps a value to a given range by truncating it to max or min.
int Clamp(int value, int min, int max) {
    if (value > max) return max;
    if (value < min) return min;
    return value;
}

// Clamps a value to a given range by "reflecting" it (i.e. a value 2 beyond the edge will be reflected 2 from the edge)
int Reflect(int value, int min, int max) {
    if (value > max) {
        const auto reflected = max + max - value;
        if (reflected < min) throw std::invalid_argument("Value is too large to be reflected");
        return reflected;
    }

    if (value < min) {
        const auto reflected = min + min - value;
        if (reflected > max) throw std::invalid_argument("Value is too small to be reflected");
        return reflected;
    }

    return value;
}


}