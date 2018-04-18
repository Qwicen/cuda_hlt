/** @file SOAField.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2017-10-03
 *
 * @brief convenient SOA fields: the ugly mechanics that work under the hood
 */

#ifndef SOAFIELD_H
#define SOAFIELD_H

#include "SOATypelistUtils.h"

/// namespace to encapsulate SOA stuff
namespace SOA {
    /// implementation details for convenient SOA fields
    namespace impl {
        // forward decl.
        struct dummy;

        /** @brief class from which a field tag derives
         *
         * @tparam T            type of the field
         * @tparam FIELD        field (child) type to use
         *
         * @author Manuel Schiller <Manuel.Schiller@cern.ch>
         * @date 2017-10-03
         */
        template <typename T, class FIELD>
        class FieldBase {
            public:
                // make sure that the user doesn't do strange things...
                static_assert(!SOA::Typelist::is_wrapped<T>::value,
                        "Error: Do no nest FieldBase<FieldBase<T>, FIELD>!");
                /// the field tag itself
                using self_type = FIELD;
                /// tell Container/View that we wrap a type
                using wrap_tag = struct {};
                /// type we wrap
                using type = T;
            protected:
                /** @brief low-level struct for as base class for accessors
                 *
                 * @tparam SKIN skin which this accessor will use
                 *
                 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
                 * @date 2017-10-03
                 */
                template <class SKIN, std::size_t IDX, class BASE>
                struct AccessorBase {
                    /// retrieve reference to the field
                    decltype(std::declval<BASE&>().template get<IDX>())
                    _get() noexcept(noexcept(
                            std::declval<BASE&>().template get<IDX>()))
                    { return static_cast<SKIN&>(*this).template get<IDX>(); }
                    /// retrieve const reference to the field
                    decltype(std::declval<const BASE&>().template get<IDX>())
                    _get() const noexcept(noexcept(
                            std::declval<const BASE&>().template get<IDX>()))
                    {
                        return static_cast<const SKIN&>(
                                *this).template get<FIELD>();
                    }
                };
        };
    } // impl
} // SOA

/** @brief define the standard accessors for use in a field
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2017-10-03
 *
 * @note The name of the defined field struct starts with "f_" by convention,
 * please adhere to that convention.
 *
 * @param accessorname  name the accessor routines will get
 *
 * See the SOAFIELD macro below for an example of this macro's usage.
 */
#define SOAFIELD_ACCESSORS(accessorname) \
    auto accessorname() noexcept(noexcept(\
                std::declval<accessors&>()._get())) -> decltype(std::declval<accessors&>()._get())\
    { return this->_get(); } \
    auto accessorname() const noexcept(noexcept( \
                std::declval<const accessors&>()._get())) -> decltype(std::declval<const accessors&>()._get())\
    { return this->_get(); }

/** @brief define a field with name name and type type, with custom accessors
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2017-10-03
 *
 * @note The name of the defined field struct starts with "f_" by convention,
 * please adhere to that convention.
 *
 * @param name  name of the field struct
 * @param type  data type "contained" in the field
 * @param ...   body of the accessor struct
 *
 * This is how one would define an integer field which contains flags, along
 * with a couple of routines to test/set those flags:
 * @code
 * SOAFIELD(f_flags, int,
 *     SOAFIELD_ACCESSORS(flags) // default accessors: flags()
 *     enum Flag { Used = 0x1, Dead = 0x2 };
 *     bool isUsed() const { return flags() & Used; }
 *     bool isDead() const { return flags() & Dead; }
 *     bool setUsed(bool newState = true)
 *     {
 *         int retVal = flags();
 *         flags() = (retVal & ~Used) | (-newState & Used);
 *         return retVal & Used;
 *     }
 *     bool setDead(bool newState = true)
 *     {
 *         int retVal = flags();
 *         flags() = (retVal & ~Dead) | (-newState & Dead);
 *         return retVal & Dead;
 *     }
 *     void printflags() { std::printf("flags: %08x\n", flags()); }
 * );
 * @endcode
 */
#define SOAFIELD(name, type, ... /* body */) \
    struct name : SOA::impl::FieldBase<type, name> { \
        template <class SKIN, class TL, class BASE> struct accessors : AccessorBase<SKIN, TL::template find<name>(), BASE> \
        { __VA_ARGS__ }; }
/** @brief define a field with name f_name, accessors named name, of type type
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2017-10-03
 *
 * @note The name of the defined field struct starts with "f_" by convention,
 * please adhere to that convention.
 *
 * @param name          name of the field struct
 * @param type          data type "contained" in the field
 * @param accessorname  name of the accessor function (getter/setter)
 *
 * To define a field f_x which contains a float, with standard getters/setters
 * called x() that return a (const) float&, one would write this code:
 * @code
 * SOAFIELD_TRIVIAL(f_x, x, float);
 * @endcode
 */
#define SOAFIELD_TRIVIAL(name, accessorname, type) \
    SOAFIELD(name, type, SOAFIELD_ACCESSORS(accessorname))

#endif // SOAFIELD_H

// vim: sw=4:tw=78:ft=cpp:et
