#include <signal.h>

namespace alphaone
{

template <typename T>
class Wrapper
{
  public:
    static Wrapper &GetInstance()
    {
        static Wrapper w = Wrapper();
        return w;
    }
    void Set(T *t)
    {
        t_ = t;
    }
    T *Get()
    {
        return t_;
    }

  private:
    Wrapper() = default;
    T *t_;
};


}  // namespace alphaone
