int index(int x, int y, int2 shape){
    return x * shape.y + y;
}

int index(int2 p, int2 shape){
    return p.x * shape.y + p.y;
}

bool advanceIterator(int2& pos, int2 shape){
    pos.y++;
    if (pos.y < shape.y)
        return true;

    pos.y = 0;
    pos.x++;
    if (pos.x < shape.x)
        return true;

    return false;
}

int totalSize(int2 shape){
    return shape.x * shape.y;
}

int totalSize(int3 shape){
    return shape.x * shape.y * shape.z;
}