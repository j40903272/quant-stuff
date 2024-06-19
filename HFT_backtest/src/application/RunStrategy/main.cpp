#include "strategy/Main.h"

#include <boost/program_options.hpp>

namespace ao = alphaone;
namespace bp = boost::program_options;

namespace alphaone
{
// REGISTER_STRATEGY(CrossExchangeArbitrageStrategy)
REGISTER_STRATEGY(TestMakerStrategy)
}  // namespace alphaone

int main(int argc, char **argv)
{
    spdlog::set_pattern("[%^%=8l%$] %v");
    std::cout << alphaone::Version::GetVersionInfo() << '\n';

    // parse arguments
    // clang-format off
    bp::options_description descriptions("RunStrategy");
    descriptions.add_options()
        ("configuration,c", bp::value<std::vector<std::string>>()->multitoken(), "configurations")
        ("date,d", bp::value<int32_t>()->default_value(20200526), "date")
        ("fitter,f", bp::value<std::string>()->default_value(""), "fitter json result")
        ("production", "production mode otherwise simulation mode by default")
        ("help,h", "help")
        ("log_level,l", bp::value<int>()->default_value(2),"spdlog level") ;
    // clang-format on
    boost::program_options::variables_map vm;
    try
    {
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, descriptions), vm);
        if (vm.count("help"))
        {
            std::cout << descriptions << '\n';
            exit(1);
        }
        boost::program_options::notify(vm);
    }
    catch (const boost::program_options::error &e)
    {
        std::cout << "ERROR: " << e.what() << '\n';
        std::cout << descriptions << '\n';
        exit(1);
    }

    ao::Main main{vm};
    main.Execute();

    return 0;
}