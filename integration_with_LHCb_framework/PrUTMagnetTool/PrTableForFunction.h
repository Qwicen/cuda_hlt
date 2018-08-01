//This code was extracted from the LHCb experiment's LHCb project at https://gitlab.cern.ch/lhcb/Rec/blob/master/Pr/PrVeloUT/src/PrTableForFunction.h

// $Id: PrTableForFunction.h,v 1.3 2009-10-30 13:19:13 wouter Exp $

#ifndef PRTABLEFORFUNCTION_H
#define PRTABLEFORFUNCTION_H 1

#include <string>
#include <vector>




  /** @class PrTableForFunction PrTableForFunction.h
   *
   *  Some internal class for pat
   *
   * @code
   * Example:
   *
   * Fill table
   * ----------
   * addVariable(10,-5.,5.);
   * addVariable(20,-50.,50.);
   *
   * resetIndexVector();
   * std::vector<float> var;
   * int iover = 0;
   * while(!iover) {
   *  getVariableVector(var);
   *  fillTable( fun (var[0],var[1]) );
   *  iover= incrementIndexVector();
   * }
   *
   * Retrieve information
   * --------------------
   * var[0]=1.5;
   * var[1]=4.1;
   * getValueFromTable(var);
   * or
   * getInterpolatedValueFromTable(var);
   * @endcode
   *
   *  @author Mariusz Witek
   *  @date   2006-09-25
   *  @update for A-Team framework 2007-08-20 SHM
   *
   */

  class PrTableForFunction  {
  public:

  

    /// Standard constructor
    PrTableForFunction( const std::string& type,
                      const std::string& name);

    virtual ~PrTableForFunction( ); ///< Destructor

    void   addVariable(int nBin, float lowVal, float highVal);
    void   prepareTable();
    void   resetIndexVector();
    int    incrementIndexVector();
    void   fillTable(float lutValue);
    float getVariable(int ivar);
    void   getVariableVector(std::vector<float>& var);
    float getValueFromTable(std::vector<float>& var);
    float getInterpolatedValueFromTable(std::vector<float>& var);

    void clear() ;

    inline std::vector<float> returnTable(){
      return m_table;
    }
    


  protected:

  private:

    void   createTable();
    void   deleteTable();
    void   resetVariableVector();
    int    tableLocation();
    void   calculateVariableVector();
    void   calculateIndexVector(std::vector<float>& var);
    void   calculateClosestIndexVector(std::vector<float>& var);

    int     m_nVar;
    std::vector<int>    m_nPointVar;
    std::vector<float> m_minVar;
    std::vector<float> m_maxVar;
    std::vector<float> m_deltaVar;
    std::vector<float> m_table;

    std::vector<int>    m_indexVector;
    std::vector<float> m_variableVector;

  };

#endif // TABLEFORFUNCTION_H
