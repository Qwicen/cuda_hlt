![SOA Container logo](doc/SOAContainer.svg)
# SOAContainer

SOAContainer is a class that mimics the interface of std::vector as much
as possible. In fact, from a user's point of view, there is not too much
difference between a `SOA::Container` and an array of structure (AOS)
`std::vector<Hit>` where Hit is an old-style class/structure that we all
know. You can sort, push_back, emplace, insert, erase, reserve, size,
index and get iterators from a `SOA::Container` just like you would for the
AOS case. Moreover, there's the `SOA::View` class which does the same
forward fixed-size ranges.

You can read the all the details in the [tutorial](./tutorial/tutorial.md).

To cut a long story short, SOAContainer gives you containers with SOA
storage layout which requires very little hassle for the designer of a
container, and virtually no change to code using that container. The
following little bit of example code (which defines a container of 3D points
in cartesian coordinates, and converts it into cylindrical coordinates) may
give you a flavour of what's possible:

```c++
// SOAContainer: define fields (data), skin (interface to modify data)
namespace XYZPoint {
    // field called x, with getters/setters x(), of type float
    SOAFIELD_TRIVIAL(x, x, float);
    SOAFIELD_TRIVIAL(y, y, float); // ditto, for y
    SOAFIELD_TRIVIAL(z, z, float); // ditto, for z
    SOASKIN_TRIVIAL(Skin, x, y, z); // uses fields x, y, z
};
// define a SOA container - mostly has std::vector's interface, so is
// familiar
SOA::Container<std::vector, Point::Skin> c = /* fill somehow... */;

// we now have a big vector of points in 3D, cartesian coordinates...
// get the x/y portion of it

// get a view into the container which contains only two fields: x and y
auto xyview = c.view<XYZPoint::x, XYZPoint::y>();

// let's see how we can convert them to cylindrical coordinates in a nice
// manner
namespace RPhiPoint { // point in polar coordinates in xy plane
    SOAFIELD_TRIVIAL(r, r, float);
    SOAFIELD_TRIVIAL(phi, phi, float);
    SOASKIN_TRIVIAL(Skin, r, phi);
};
// container to hold the converted r and phi quantities
SOAContainer<std::vector, RPhiPoint::Skin> crphi;
crphi.reserve(xyview.size());
for (auto xy: xyview) { // calculate r and phi
    crphi.emplace_back(
        std::sqrt(xy.x() * xy.x() + xy.y() * xy.y()),
        std::atan2(xy.y(), xy.x()));
}

// now we want a view in cylindrical coordinates...
auto zview = c.view<XYZPoint::z>(); // view of z alone
// zip together two views to form a new one: (r, phi, z)
auto view_cyl = zip(crphi, zview);
// you get back a normal view, and can do things with it:
for (auto p: view_cyl) {
    std::cout << "Point r " << p.r() << " phi " << p.phi() << " z " <<
        p.z() << std::endl;
}
```
